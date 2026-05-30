package llm

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// ProviderKeys holds API credentials for the LLM providers.
type ProviderKeys struct {
	Gemini     string
	OpenRouter string
	NVIDIANim  string
}

// ProviderURLs holds API endpoints for the LLM providers.
type ProviderURLs struct {
	OpenRouter string
	NVIDIANim  string
	VLLM       string
}

// NewClientFromProvider instantiates an LLMClient based on the specified provider and model.
// TODO: Support dynamic custom timeout overrides per model in ProviderURLs.
func NewClientFromProvider(provider, model string, keys ProviderKeys, urls ProviderURLs, timeout time.Duration) (LLMClient, error) {
	switch provider {
	case "gemini":
		if keys.Gemini == "" {
			return nil, fmt.Errorf("gemini: GEMINI_API_KEY is required (model %s)", model)
		}
		return NewGeminiClient(keys.Gemini, model, timeout), nil

	case "openrouter":
		if keys.OpenRouter == "" {
			return nil, fmt.Errorf("openrouter: OPENROUTER_API_KEY is required (model %s)", model)
		}
		return NewOpenRouterClient(keys.OpenRouter, urls.OpenRouter, model, timeout), nil

	case "nvidia-nim":
		if keys.NVIDIANim == "" {
			return nil, fmt.Errorf("nvidia-nim: NVIDIA_NIM_API_KEY is required (model %s)", model)
		}
		return NewNIMClient(keys.NVIDIANim, urls.NVIDIANim, model, timeout), nil

	case "local":
		// Dynamic endpoint routing for local vLLM servers based on normalized model names.
		// e.g. VLLM_MICROSOFT_PHI_4_MINI_INSTRUCT_URL overrides default urls.VLLM
		envKey := "VLLM_" + normalizeModelName(model) + "_URL"
		vllmURL := os.Getenv(envKey)
		if vllmURL == "" {
			vllmURL = urls.VLLM
		}
		if vllmURL == "" {
			vllmURL = "http://vllm-inference:8001/v1/chat/completions"
		}
		return NewVLLMClient(vllmURL, model, timeout), nil

	default:
		return nil, fmt.Errorf("unknown provider %q for model %s (valid: gemini, openrouter, nvidia-nim, local)", provider, model)
	}
}

// normalizeModelName converts a model identifier to an uppercase alphanumeric env-safe string.
func normalizeModelName(model string) string {
	var r []rune
	for _, ch := range model {
		if (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') {
			r = append(r, ch)
		} else {
			r = append(r, '_')
		}
	}
	return strings.ToUpper(strings.Trim(string(r), "_"))
}
