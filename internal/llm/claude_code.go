package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

const (
	claudeCodeProvider       = "claude-code"
	claudeCodeDefaultCommand = "claude"
	claudeCodeCommandEnvVar  = "JOBSCOUT_CLAUDE_PATH"
	claudeCodeJSONSuffix     = "Respond with only valid JSON. Do not wrap it in code fences or add commentary."
)

// claudeCodeLLM runs prompts through the locally installed Claude Code CLI in
// non-interactive print mode, so requests use the user's Claude Code login
// instead of an API key.
type claudeCodeLLM struct {
	command string
	model   string
}

type claudeCodeResult struct {
	Type    string `json:"type"`
	Subtype string `json:"subtype"`
	IsError bool   `json:"is_error"`
	Result  string `json:"result"`
	Usage   struct {
		InputTokens              int `json:"input_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		OutputTokens             int `json:"output_tokens"`
	} `json:"usage"`
}

func newClaudeCodeLLM(modelName string) (*claudeCodeLLM, error) {
	command := strings.TrimSpace(os.Getenv(claudeCodeCommandEnvVar))
	if command == "" {
		command = claudeCodeDefaultCommand
	}
	resolved, err := exec.LookPath(command)
	if err != nil {
		return nil, fmt.Errorf("claude CLI not found (install Claude Code and run `claude` once to log in, or set %s): %w", claudeCodeCommandEnvVar, err)
	}
	return &claudeCodeLLM{command: resolved, model: strings.TrimSpace(modelName)}, nil
}

func (m *claudeCodeLLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	var opts llms.CallOptions
	for _, option := range options {
		option(&opts)
	}

	system, prompt := claudeCodePrompt(messages)
	if opts.JSONMode {
		system = strings.TrimSpace(system + "\n\n" + claudeCodeJSONSuffix)
	}
	if strings.TrimSpace(prompt) == "" {
		return nil, fmt.Errorf("claude-code: empty prompt")
	}

	args := []string{
		"-p",
		"--output-format", "json",
		"--tools", "",
		"--strict-mcp-config",
		"--no-session-persistence",
	}
	if model := strings.TrimSpace(opts.Model); model != "" {
		args = append(args, "--model", model)
	} else if m.model != "" {
		args = append(args, "--model", m.model)
	}
	if system != "" {
		args = append(args, "--system-prompt", system)
	}

	cmd := exec.CommandContext(ctx, m.command, args...)
	// Run outside the project so Claude Code does not pick up a CLAUDE.md.
	cmd.Dir = os.TempDir()
	cmd.Stdin = strings.NewReader(prompt)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	logDebug("claude-code: running model=%s prompt_chars=%d", m.model, len(prompt))
	runErr := cmd.Run()

	var result claudeCodeResult
	if err := json.Unmarshal(bytes.TrimSpace(stdout.Bytes()), &result); err != nil {
		if runErr != nil {
			return nil, fmt.Errorf("claude-code: %v: %s", runErr, claudeCodeErrorText(stdout.String(), stderr.String()))
		}
		return nil, fmt.Errorf("claude-code: could not parse CLI output: %v", err)
	}
	if result.IsError || runErr != nil {
		return nil, fmt.Errorf("claude-code: %s", claudeCodeErrorText(result.Result, stderr.String()))
	}

	content := result.Result
	if opts.JSONMode {
		content = stripClaudeCodeFences(content)
	}
	inputTokens := result.Usage.InputTokens + result.Usage.CacheCreationInputTokens + result.Usage.CacheReadInputTokens
	return &llms.ContentResponse{Choices: []*llms.ContentChoice{{
		Content:    content,
		StopReason: result.Subtype,
		GenerationInfo: map[string]any{
			"InputTokens":  inputTokens,
			"OutputTokens": result.Usage.OutputTokens,
		},
	}}}, nil
}

func (m *claudeCodeLLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, m, prompt, options...)
}

// claudeCodePrompt splits langchaingo messages into a system prompt and a
// single stdin prompt. Multi-turn histories are flattened into a transcript.
func claudeCodePrompt(messages []llms.MessageContent) (string, string) {
	var system []string
	type turn struct {
		role string
		text string
	}
	var turns []turn
	for _, message := range messages {
		text := claudeCodeMessageText(message)
		if text == "" {
			continue
		}
		switch message.Role {
		case llms.ChatMessageTypeSystem:
			system = append(system, text)
		case llms.ChatMessageTypeAI:
			turns = append(turns, turn{role: "Assistant", text: text})
		default:
			turns = append(turns, turn{role: "User", text: text})
		}
	}

	if len(turns) == 1 {
		return strings.Join(system, "\n\n"), turns[0].text
	}
	parts := make([]string, 0, len(turns))
	for _, t := range turns {
		parts = append(parts, t.role+": "+t.text)
	}
	return strings.Join(system, "\n\n"), strings.Join(parts, "\n\n")
}

func claudeCodeMessageText(message llms.MessageContent) string {
	var parts []string
	for _, part := range message.Parts {
		if text, ok := part.(llms.TextContent); ok && strings.TrimSpace(text.Text) != "" {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func stripClaudeCodeFences(content string) string {
	trimmed := strings.TrimSpace(content)
	if !strings.HasPrefix(trimmed, "```") || !strings.HasSuffix(trimmed, "```") {
		return content
	}
	trimmed = strings.TrimSuffix(trimmed, "```")
	if idx := strings.Index(trimmed, "\n"); idx >= 0 {
		trimmed = trimmed[idx+1:]
	} else {
		trimmed = strings.TrimPrefix(trimmed, "```")
	}
	return strings.TrimSpace(trimmed)
}

func claudeCodeErrorText(primary string, stderr string) string {
	for _, text := range []string{primary, stderr} {
		if text = strings.TrimSpace(text); text != "" {
			if len(text) > 500 {
				text = text[:500] + "..."
			}
			return text
		}
	}
	return "unknown error"
}

func claudeCodeModelOptions() []string {
	return modelOptionsForProvider(claudeCodeProvider)
}
