export type JagXMessage = { role: 'system' | 'user' | 'assistant' | 'tool'; content: string };

export type ChatRequest = {
  messages: JagXMessage[];
  model?: string;
  temperature?: number;
  max_new_tokens?: number;
};

export type ChatResponse = {
  message: JagXMessage;
  model?: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
};

export class JagXApiClient {
  constructor(private readonly baseUrl: string) {}

  async health(): Promise<{ status: string }> {
    const response = await fetch(`${this.baseUrl.replace(/\/$/, '')}/health`);
    if (!response.ok) throw new Error(`JagX health request failed: ${response.status}`);
    return response.json() as Promise<{ status: string }>;
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl.replace(/\/$/, '')}/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`JagX chat request failed: ${response.status}`);
    return response.json() as Promise<ChatResponse>;
  }
}
