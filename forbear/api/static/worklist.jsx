/* The merchant's daily worklist. Calm, plain-language, no ML jargon.

   Everything on screen came from GET /worklist, which is a database read of a
   plan computed ahead of time - there is no "loading the model" moment here,
   only "loading the list". */

const { useState, useEffect, useCallback } = React;

const rupees = (n) => "₹" + Math.round(n || 0).toLocaleString("en-IN");

function uuid() {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

const ACTION_LABEL = {
  send_payment_link: "Send payment link",
  retry: "Retry payment",
  update_card: "Ask to update card",
};

function WhyLink({ reason }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="why">
      <button className="why-toggle" onClick={() => setOpen(!open)}>
        {open ? "hide" : "why?"}
      </button>
      {open && <p className="why-text">{reason}</p>}
    </div>
  );
}

function ChaseRow({ row, onDone }) {
  const [state, setState] = useState("idle"); // idle | sending | done | error

  const act = useCallback(async () => {
    setState("sending");
    try {
      const res = await fetch(`/actions/${row.record_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: row.action,
          idempotency_key: uuid(),
        }),
      });
      if (!res.ok) throw new Error("failed");
      setState("done");
      onDone(row.record_id);
    } catch (e) {
      setState("error");
    }
  }, [row, onDone]);

  return (
    <div className="row">
      <div className="row-main">
        <div className="row-name">{row.customer_name}</div>
        <div className="row-amount">{rupees(row.amount_rupees)}</div>
      </div>
      <div className="row-reason">{row.reason}</div>
      <div className="row-action">
        {state === "done" ? (
          <span className="done-badge">Done today ✓</span>
        ) : (
          <button
            className="primary-btn"
            onClick={act}
            disabled={state === "sending"}
          >
            {state === "sending"
              ? "Sending…"
              : ACTION_LABEL[row.action] || "Take action"}
          </button>
        )}
      </div>
      {state === "error" && (
        <div className="row-error">Something went wrong. Try again.</div>
      )}
    </div>
  );
}

function WaitRow({ row }) {
  const date = row.expected_at
    ? new Date(row.expected_at).toLocaleDateString("en-IN", {
        day: "numeric",
        month: "short",
      })
    : "soon";
  return (
    <div className="row row-quiet">
      <div className="row-main">
        <div className="row-name">{row.customer_name}</div>
        <div className="row-amount">{rupees(row.amount_rupees)}</div>
      </div>
      <div className="row-reason">Expected around {date}.</div>
    </div>
  );
}

function LeaveAloneRow({ row }) {
  return (
    <div className="row row-quiet">
      <div className="row-main">
        <div className="row-name">{row.customer_name}</div>
        <div className="row-amount">{rupees(row.amount_rupees)}</div>
      </div>
      <div className="row-reason">{row.reason}</div>
    </div>
  );
}

function ProtectedRow({ row }) {
  return (
    <div className="row row-quiet">
      <div className="row-main">
        <div className="row-name">{row.customer_name}</div>
        <div className="row-amount protected-amount">
          {rupees(row.value_protected_rupees)} protected
        </div>
      </div>
      <div className="row-reason">{row.reason}</div>
    </div>
  );
}

function ProtectedPanel({ protectedData }) {
  if (!protectedData) return null;

  if (!protectedData.available) {
    return (
      <p className="protected-note">
        Customers saved by leaving them alone: {protectedData.message}
      </p>
    );
  }

  if (protectedData.customers_protected === 0) return null;

  return (
    <section className="section protected-panel">
      <div className="protected-header">
        <p className="protected-sentence">{protectedData.sentence}</p>
      </div>
      <div className="section-body">
        {protectedData.saves.map((row) => (
          <ProtectedRow key={row.record_id} row={row} />
        ))}
      </div>
    </section>
  );
}

function Section({ title, count, defaultOpen, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="section">
      <button className="section-header" onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span className="section-count">{count}</span>
        <span className="chevron">{open ? "▾" : "▸"}</span>
      </button>
      {open && <div className="section-body">{children}</div>}
    </section>
  );
}

function App() {
  const [data, setData] = useState(null);
  const [protectedData, setProtectedData] = useState(null);
  const [error, setError] = useState(null);
  const [doneToday, setDoneToday] = useState(() => new Set());

  const load = useCallback(async () => {
    try {
      const res = await fetch("/worklist?date=today");
      if (!res.ok) throw new Error("failed to load worklist");
      setData(await res.json());
    } catch (e) {
      setError(String(e));
    }
    try {
      const res = await fetch("/worklist/protected");
      if (res.ok) setProtectedData(await res.json());
    } catch (e) {
      // The protected panel is a bonus view; its failure should not block
      // the worklist itself from rendering.
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const markDone = useCallback((recordId) => {
    setDoneToday((prev) => new Set(prev).add(recordId));
  }, []);

  if (error) return <div className="banner error-banner">{error}</div>;
  if (!data) return <div className="banner">Loading today's worklist…</div>;

  const chase = data.chase.filter((r) => !doneToday.has(r.record_id));
  const doneCount = data.chase.length - chase.length;

  return (
    <div className="page">
      <header className="hero">
        <p className="hero-sentence">{data.fear_number.sentence}</p>
        <a className="advanced-link" href="/advanced">
          advanced view
        </a>
      </header>

      <Section title="Chase today" count={chase.length} defaultOpen>
        {chase.length === 0 ? (
          <p className="empty">
            {doneCount > 0
              ? "All done for today."
              : "Nothing needs chasing right now."}
          </p>
        ) : (
          chase.map((row) => (
            <div key={row.record_id}>
              <ChaseRow row={row} onDone={markDone} />
              <WhyLink reason={row.reason} />
            </div>
          ))
        )}
      </Section>

      <Section title="Will pay on their own" count={data.wait.length}>
        {data.wait.length === 0 ? (
          <p className="empty">Nobody in this list today.</p>
        ) : (
          data.wait.map((row) => <WaitRow key={row.record_id} row={row} />)
        )}
      </Section>

      <Section title="Leave alone" count={data.leave_alone.length}>
        {data.leave_alone.length === 0 ? (
          <p className="empty">Nobody in this list today.</p>
        ) : (
          data.leave_alone.map((row) => (
            <LeaveAloneRow key={row.record_id} row={row} />
          ))
        )}
      </Section>

      <ProtectedPanel protectedData={protectedData} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
