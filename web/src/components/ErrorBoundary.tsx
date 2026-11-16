import { Component, type ErrorInfo, type ReactNode } from "react";

import { Button } from "@/components/ui/button";

/** A bug on one page must not blank the whole app: show what happened and a way back. */
export class ErrorBoundary extends Component<{ resetKey: string; children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("page crashed", error, info.componentStack);
  }

  componentDidUpdate(prev: { resetKey: string }) {
    // Moving to another page clears the error.
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="mx-auto flex max-w-2xl flex-col gap-3 border-l-4 border-ink pl-5">
        <h1 className="text-xl font-semibold">This page stopped working</h1>
        <p className="text-base text-graphite">
          Your documents, traces and settings are not affected. Reload the page; if it happens again, the browser
          console has the details.
        </p>
        <p className="text-sm">{this.state.error.message}</p>
        <div className="flex gap-2 pt-1">
          <Button size="sm" onClick={() => window.location.reload()}>
            Reload
          </Button>
          <Button size="sm" variant="outline" onClick={() => this.setState({ error: null })}>
            Try again
          </Button>
        </div>
      </div>
    );
  }
}
