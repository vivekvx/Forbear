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
    failure_class   failure_class NOT NULL,
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


-- Per-customer contact budget is enforced by counting rows here.
CREATE TABLE contacts (
    id          BIGSERIAL PRIMARY KEY,
    customer_id BIGINT      NOT NULL REFERENCES customers (id),
    channel     contact_channel NOT NULL,
    sent_at     TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX contacts_customer_sent_idx ON contacts (customer_id, sent_at);


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
