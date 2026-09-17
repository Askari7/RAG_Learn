import {
  ApiError,
  ChatResponse,
  ThreadMessagesResponse,
  ThreadsResponse,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

async function request<T>(
  path: string,
  init: RequestInit,
  timeoutMs: number
): Promise<T> {
  const controller = new AbortController();
  let didTimeout = false;
  const timer = setTimeout(() => {
    didTimeout = true;
    controller.abort();
  }, timeoutMs);

  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new ApiError(
        "http",
        (body && body.detail) || res.statusText,
        res.status
      );
    }

    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(
        didTimeout ? "timeout" : "connection",
        "Request aborted"
      );
    }
    if (err instanceof TypeError) {
      throw new ApiError("connection", "Could not connect to FastAPI");
    }
    throw new ApiError(
      "unknown",
      err instanceof Error ? err.message : String(err)
    );
  } finally {
    clearTimeout(timer);
  }
}

export function getThreads(limit = 50): Promise<ThreadsResponse> {
  return request<ThreadsResponse>(`/threads?limit=${limit}`, {}, 30_000);
}

export function getThreadMessages(
  threadId: string
): Promise<ThreadMessagesResponse> {
  return request<ThreadMessagesResponse>(
    `/threads/${encodeURIComponent(threadId)}/messages`,
    {},
    30_000
  );
}

export function postChat(
  question: string,
  threadId: string
): Promise<ChatResponse> {
  return request<ChatResponse>(
    "/chat",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: threadId }),
    },
    600_000
  );
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    switch (err.kind) {
      case "connection":
        return "❌ Could not connect to FastAPI.";
      case "timeout":
        return "⏱️ Request timed out.";
      case "http":
        return `❌ API error: ${err.message}`;
      default:
        return `❌ Error: ${err.message}`;
    }
  }
  return `❌ Error: ${err instanceof Error ? err.message : String(err)}`;
}
