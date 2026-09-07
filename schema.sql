-- Forbear schema.
--
-- Design rule: illegal states must be unrepresentable. Anything expressible as
-- a constraint is a constraint, not a convention. Attempt counting in
-- particular is protected by a UNIQUE key, because an over-count here becomes
-- a regulatory cap breach downstream.

CREATE TYPE mandate_status AS ENUM ('active', 'paused', 'revoked', 'expired');

CREATE TYPE failure_class AS ENUM (
    'time_dependent',
    'transient',
    'reauth_required',
    'terminal'
);

CREATE TYPE record_status AS ENUM (
    'open',
    'scheduled',
    'in_flight',
    'recovered',
    'abandoned',
    'skipped'
);

CREATE TYPE attempt_outcome AS ENUM (
    'pending',
    'success',
    'failure',
    'blocked_by_guard'
);

CREATE TYPE contact_channel AS ENUM ('payment_link', 'sms', 'email');

-- A pre-debit notification is a regulatory precondition for a debit, not a
-- dunning message. The guard has to tell them apart, so the distinction is
-- stored rather than inferred from the channel.
CREATE TYPE contact_purpose AS ENUM ('pre_debit_notification', 'dunning');


CREATE TABLE customers (
    id          BIGSERIAL PRIMARY KEY,
    external_id TEXT        NOT NULL UNIQUE,  -- Razorpay customer_id
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);


CREATE TABLE subscriptions (
    id                 BIGSERIAL PRIMARY KEY,
    customer_id        BIGINT      NOT NULL REFERENCES customers (id),
    external_id        TEXT        NOT NULL UNIQUE,  -- Razorpay subscription_id
    plan_amount        BIGINT      NOT NULL CHECK (plan_amount > 0),  -- paise
    billing_cycle_days INTEGER     NOT NULL CHECK (billing_cycle_days > 0),
    mandate_status     mandate_status NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX subscriptions_customer_idx ON subscriptions (customer_id);


CREATE TABLE at_risk_records (
    id              BIGSERIAL PRIMARY KEY,
    subscription_id BIGINT      NOT NULL REFERENCES subscriptions (id),
    customer_id     BIGINT      NOT NULL REFERENCES customers (id),
    invoice_id      TEXT        NOT NULL UNIQUE,  -- Razorpay invoice_id
    amount          BIGINT      NOT NULL CHECK (amount > 0),  -- paise
    failure_code    TEXT        NOT NULL,
    -- NULL means the classifier had no mapping for failure_code: the record is
    -- on the exception list awaiting a human, not silently defaulted to a
    -- class. Nothing downstream may treat NULL as a recoverable class.
    failure_class   failure_class,
    status          record_status NOT NULL DEFAULT 'open',
    uplift_score    DOUBLE PRECISION,
    whittle_index   DOUBLE PRECISION,
    skip_reason     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- A skip is a first-class decision, so it must carry its reason. And a
    -- reason without a skip is a lie about why the record is where it is.
    CONSTRAINT skip_requires_reason
        CHECK ((status = 'skipped') = (skip_reason IS NOT NULL))
);

CREATE INDEX at_risk_records_status_idx ON at_risk_records (status);
CREATE INDEX at_risk_records_customer_idx ON at_risk_records (customer_id);


-- Merchant worklist: the precomputed answer to "what do I do about this
-- record today". Written once by the background decisioning job that runs
-- after ingestion, read as-is by the worklist endpoint. The endpoint must
-- never compute these values itself - that is the whole latency contract - so
-- they live as columns rather than being derived at request time.
ALTER TABLE at_risk_records
    ADD COLUMN worklist_bucket TEXT
        CHECK (worklist_bucket IN ('chase', 'wait', 'leave_alone')),
    ADD COLUMN worklist_action TEXT,
    ADD COLUMN worklist_reason TEXT,
    ADD COLUMN worklist_scheduled_at TIMESTAMPTZ,
    -- For a leave_alone record: the estimated rupee cost of chasing it anyway
    -- (typically the LTV a contact would put at risk). Precomputed so an
    -- override warning is also a read, not a recomputation.
    ADD COLUMN worklist_cost_paise BIGINT,
    ADD COLUMN worklist_decided_at TIMESTAMPTZ,
    -- The raw allocator skip_reason code behind a leave_alone decision (NULL
    -- for chase/wait). Decisioning runs allocate() with commit=False, so the
    -- bare skip_reason column above (which the real commit=True cycle writes)
    -- stays empty here; this is the preview's own copy of the same code, kept
    -- so a reader can tell a do-not-disturb skip (negative_net_value - the
    -- ones a save can come from) apart from a terminal one (nothing could
    -- have been done regardless of ground truth) without parsing audit JSON.
    ADD COLUMN worklist_skip_reason TEXT;


-- Demo-only answer key: which do_not_disturb-style leave_alone decisions
-- actually protected a customer, per the synthetic generator's ground truth.
-- Real production ingestion (the webhook path) never writes this table - a
-- real webhook has no counterfactual to report - so its emptiness is exactly
-- how the worklist tells production data apart from a demo run and hides the
-- protected-customers panel rather than fabricate one.
CREATE TABLE demo_ground_truth (
    at_risk_record_id      BIGINT  PRIMARY KEY REFERENCES at_risk_records (id),
    would_churn_if_contacted  BOOLEAN NOT NULL,
    would_pay_without_contact BOOLEAN NOT NULL,
    remaining_ltv_paise       BIGINT  NOT NULL
);


-- Every merchant-triggered action, keyed for idempotency. A double-tap on the
-- worklist must never send two payment links: the unique index on
-- idempotency_key is what makes the second request a read of the first
-- request's result rather than a second send.
CREATE TABLE merchant_actions (
    id                BIGSERIAL PRIMARY KEY,
    idempotency_key   TEXT        NOT NULL UNIQUE,
    at_risk_record_id BIGINT      NOT NULL REFERENCES at_risk_records (id),
    action_type       TEXT        NOT NULL,
    is_override       BOOLEAN     NOT NULL DEFAULT false,
    result            JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX merchant_actions_record_idx ON merchant_actions (at_risk_record_id);


CREATE TABLE attempts (
    id                BIGSERIAL PRIMARY KEY,
    at_risk_record_id BIGINT      NOT NULL REFERENCES at_risk_records (id),
    attempt_number    INTEGER     NOT NULL CHECK (attempt_number > 0),
    scheduled_at      TIMESTAMPTZ NOT NULL,
    executed_at       TIMESTAMPTZ,
    outcome           attempt_outcome NOT NULL DEFAULT 'pending',
    guard_verdict     JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- The attempt cap is counted off this key. Two rows claiming the same
    -- attempt number would under-count consumed attempts.
    UNIQUE (at_risk_record_id, attempt_number),

    -- A settled attempt has an execution time; an unsettled one does not.
    CONSTRAINT settled_attempt_has_executed_at
        CHECK ((outcome = 'pending') = (executed_at IS NULL)),

    -- A guard block is only meaningful with the verdict that caused it.
    CONSTRAINT block_requires_verdict
        CHECK (outcome <> 'blocked_by_guard' OR guard_verdict IS NOT NULL)
);

CREATE INDEX attempts_record_idx ON attempts (at_risk_record_id);


-- Per-customer contact budget is enforced by counting rows here. The guard
-- also reads this table to confirm a pre-debit notification exists for the
-- subscription about to be debited.
CREATE TABLE contacts (
    id              BIGSERIAL PRIMARY KEY,
    customer_id     BIGINT      NOT NULL REFERENCES customers (id),
    -- Nullable: a customer-level contact need not target one subscription.
    subscription_id BIGINT      REFERENCES subscriptions (id),
    channel         contact_channel NOT NULL,
    purpose         contact_purpose NOT NULL,
    sent_at         TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- A pre-debit notification authorises a debit on one specific mandate.
    -- Without a subscription it authorises nothing.
    CONSTRAINT notification_targets_a_subscription
        CHECK (purpose <> 'pre_debit_notification' OR subscription_id IS NOT NULL)
);

CREATE INDEX contacts_customer_sent_idx ON contacts (customer_id, sent_at);
CREATE INDEX contacts_notification_idx
    ON contacts (subscription_id, purpose, sent_at);


-- Hash-linked audit chain. Entries are linked per entity: hash covers
-- prev_hash, so editing or deleting any entry breaks every later link.
CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    entity_type TEXT        NOT NULL,
    entity_id   TEXT        NOT NULL,
    action      TEXT        NOT NULL,
    details     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    prev_hash   TEXT        NOT NULL,
    hash        TEXT        NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX audit_log_entity_idx ON audit_log (entity_type, entity_id, id);


-- UNIQUE on event_id is the replay detection: a redelivered Razorpay webhook
-- collides here instead of being processed twice.
CREATE TABLE webhook_events (
    id          BIGSERIAL PRIMARY KEY,
    event_id    TEXT        NOT NULL UNIQUE,
    event_type  TEXT        NOT NULL,
    payload     JSONB       NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
