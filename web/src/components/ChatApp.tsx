"use client";

import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { errorMessage, getThreadMessages, getThreads, postChat } from "@/lib/api";
import { Message, ThreadSummary } from "@/lib/types";
import Sidebar from "./Sidebar";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import styles from "./ChatApp.module.css";

function newThreadId(): string {
  return crypto.randomUUID();
}

function setThreadIdInUrl(threadId: string) {
  const url = new URL(window.location.href);
  url.searchParams.set("thread_id", threadId);
  window.history.replaceState(null, "", url.toString());
}

export default function ChatApp() {
  const searchParams = useSearchParams();
  const fromUrl = searchParams.get("thread_id");

  const [threadId, setThreadId] = useState<string>(() => fromUrl || newThreadId());
  const [messages, setMessages] = useState<Message[]>([]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [status, setStatus] = useState<"idle" | "restoring" | "sending">(
    () => (fromUrl ? "restoring" : "idle")
  );

  const initialized = useRef(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // One-time init: write the resolved thread_id into the URL, restore its
  // history if it came from the URL, and load the sidebar's thread list.
  // Guarded against React 18 Strict Mode's dev-only double-invoke of effects.
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    setThreadIdInUrl(threadId);

    if (fromUrl) {
      getThreadMessages(threadId)
        .then((res) => setMessages(res.messages))
        .catch(() => setMessages([]))
        .finally(() => setStatus("idle"));
    }

    refreshThreads();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function refreshThreads() {
    getThreads()
      .then((res) => setThreads(res.threads))
      .catch(() => {
        /* sidebar list is a nice-to-have; ignore failures silently */
      });
  }

  function handleNewChat() {
    const id = newThreadId();
    setThreadId(id);
    setThreadIdInUrl(id);
    setMessages([]);
  }

  function handleSwitchThread(id: string) {
    setStatus("restoring");
    getThreadMessages(id)
      .then((res) => {
        setThreadId(id);
        setThreadIdInUrl(id);
        setMessages(res.messages);
      })
      .catch((err) => {
        setMessages([{ role: "assistant", content: errorMessage(err) }]);
        setThreadId(id);
        setThreadIdInUrl(id);
      })
      .finally(() => setStatus("idle"));
  }

  function handleSend(question: string) {
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setStatus("sending");

    postChat(question, threadId)
      .then((res) => {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: res.response },
        ]);
      })
      .catch((err) => {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: errorMessage(err) },
        ]);
      })
      .finally(() => {
        setStatus("idle");
        refreshThreads();
      });
  }

  const busy = status !== "idle";

  return (
    <div className={styles.layout}>
      <Sidebar
        threads={threads}
        currentThreadId={threadId}
        disabled={busy}
        onNewChat={handleNewChat}
        onSwitchThread={handleSwitchThread}
      />
      <main className={styles.main}>
        <header className={styles.header}>
          <h1 className={styles.title}>🤖 GenAI Assistant</h1>
          <p className={styles.caption}>Powered by FastAPI</p>
        </header>
        <div className={styles.messages}>
          {messages.map((message, i) => (
            <ChatMessage key={i} role={message.role} content={message.content} />
          ))}
          {status === "sending" && (
            <div className={styles.thinking}>Thinking...</div>
          )}
          <div ref={bottomRef} />
        </div>
        <ChatInput disabled={busy} onSend={handleSend} />
      </main>
    </div>
  );
}
