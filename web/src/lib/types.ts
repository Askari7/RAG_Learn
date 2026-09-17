export type Role = "user" | "assistant";

export interface Message {
  role: Role;
  content: string;
}

export interface ThreadSummary {
  thread_id: string;
  preview: string;
}

export interface ChatResponse {
  response: string;
}

export interface ThreadsResponse {
  threads: ThreadSummary[];
}

export interface ThreadMessagesResponse {
  messages: Message[];
}

export type ApiErrorKind = "connection" | "timeout" | "http" | "unknown";

export class ApiError extends Error {
  kind: ApiErrorKind;
  status?: number;

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}
