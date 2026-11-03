package llm

import (
	"testing"

	"github.com/tmc/langchaingo/llms"
)

func TestClaudeCodePromptSplitsSystemAndUser(t *testing.T) {
	messages := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are a JSON API."),
		llms.TextParts(llms.ChatMessageTypeHuman, "Return {}"),
	}
	system, prompt := claudeCodePrompt(messages)
	if system != "You are a JSON API." {
		t.Fatalf("system = %q", system)
	}
	if prompt != "Return {}" {
		t.Fatalf("prompt = %q", prompt)
	}
}

func TestClaudeCodePromptFlattensMultiTurn(t *testing.T) {
	messages := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeHuman, "first"),
		llms.TextParts(llms.ChatMessageTypeAI, "reply"),
		llms.TextParts(llms.ChatMessageTypeHuman, "second"),
	}
	_, prompt := claudeCodePrompt(messages)
	want := "User: first\n\nAssistant: reply\n\nUser: second"
	if prompt != want {
		t.Fatalf("prompt = %q, want %q", prompt, want)
	}
}

func TestStripClaudeCodeFences(t *testing.T) {
	cases := map[string]string{
		"```json\n{\"ok\":true}\n```": `{"ok":true}`,
		"```\n[1,2]\n```":             `[1,2]`,
		`{"ok":true}`:                 `{"ok":true}`,
	}
	for input, want := range cases {
		if got := stripClaudeCodeFences(input); got != want {
			t.Errorf("stripClaudeCodeFences(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestClaudeCodeProviderNeedsNoAuth(t *testing.T) {
	cfg := defaultAppConfig()
	cfg.LLM.Enabled = true
	cfg.LLM.Provider = claudeCodeProvider
	name, providerCfg, ok := effectiveLLMProvider(&cfg)
	if !ok || name != claudeCodeProvider {
		t.Fatalf("effective provider = %q ok=%v", name, ok)
	}
	if !providerCfg.Auth.None {
		t.Fatalf("expected claude-code auth to be none, got %+v", providerCfg.Auth)
	}
	if providerCfg.Model != "haiku" {
		t.Fatalf("model = %q, want haiku", providerCfg.Model)
	}
}
