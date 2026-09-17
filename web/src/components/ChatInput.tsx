"use client";

import { FormEvent, useState } from "react";
import styles from "./ChatInput.module.css";

interface ChatInputProps {
  disabled: boolean;
  onSend: (question: string) => void;
}

export default function ChatInput({ disabled, onSend }: ChatInputProps) {
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      <input
        className={styles.input}
        type="text"
        placeholder="Ask something..."
        value={value}
        disabled={disabled}
        onChange={(e) => setValue(e.target.value)}
      />
      <button
        className={styles.button}
        type="submit"
        disabled={disabled || !value.trim()}
      >
        Send
      </button>
    </form>
  );
}
