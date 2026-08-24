# Project: Forbear — recovery decisioning engine

## What this is
A system that decides which failed subscription payments are worth
chasing. Razorpay retries on a fixed T+1/T+2/T+3 schedule, then
halts and never auto-charges the invoice again. Forbear replaces
that with an allocator that spends a capped attempt budget only
where contacting the customer actually changes the outcome.

## Core thesis
Optimise for net customer value, not recovery rate. Some records
are deliberately skipped because chasing them costs more in churn
risk than the invoice is worth. The skip is a first-class decision
with a reason code and an audit entry.

## Architecture invariants
1. No LLM output may reach the payment execution path. Models
   score; deterministic code decides and executes.
2. The allocator plans; the guard permits. They share no code.
   The guard independently re-validates every action immediately
   before execution.
3. All time comparisons use server_now(). Never client timestamps.
4. Every decision including every skip writes an audit entry with
   a hash link to the previous entry.
5. Attempt caps and regulatory constraints are hard limits, never
   soft preferences.

## Stack
Python 3.11, FastAPI, PostgreSQL, pytest. No ORM — explicit SQL
for anything touching money or attempt counts.

## Naming
Single underscore_case for files and functions. No abbreviations
except established ones (LTV, CATE, AUUC).
