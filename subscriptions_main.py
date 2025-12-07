# subscriptions_main.py

import os
from datetime import datetime, timezone
from typing import Optional

import stripe
from fastapi import APIRouter, Request, Header, HTTPException
from google.cloud import firestore
from google.api_core.datetime_helpers import DatetimeWithNanoseconds

# ─────────────────────────────────────────────
# 1. Router only (NO FastAPI app here)
# ─────────────────────────────────────────────

router = APIRouter()

# Stripe keys – set these in Cloud Run env vars
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY
else:
    print("⚠️ STRIPE_API_KEY is not set – Stripe features disabled.")

# Firestore client (reuse Cloud Run service account)
db = firestore.Client()

# ─────────────────────────────────────────────
# Helper: timestamp converter
# ─────────────────────────────────────────────

def ts_or_none(unix_ts: Optional[int]) -> Optional[DatetimeWithNanoseconds]:
    if unix_ts is None:
        return None
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)

# ─────────────────────────────────────────────
# Helper: upsert subscription doc
# ─────────────────────────────────────────────

def upsert_subscription_for_uid(
    firebase_uid: str,
    stripe_customer_id: Optional[str],
    stripe_subscription_id: Optional[str],
    plan: Optional[str],
    status: str,
    trial_end: Optional[int] = None,
    current_period_end: Optional[int] = None,
    canceled_at: Optional[int] = None,
):
    is_active = status in ("trialing", "active")

    doc_ref = db.collection("subscriptions").document(firebase_uid)
    doc_ref.set(
        {
            "stripeCustomerId": stripe_customer_id,
            "stripeSubscriptionId": stripe_subscription_id,
            "plan": plan,
            "status": status,
            "isActive": is_active,
            "trialEnd": ts_or_none(trial_end),
            "currentPeriodEnd": ts_or_none(current_period_end),
            "canceledAt": ts_or_none(canceled_at),
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    print(f"✅ Updated subscription for uid={firebase_uid} → {status}")

# ─────────────────────────────────────────────
# Stripe webhook
# ─────────────────────────────────────────────

@router.post("/stripe-webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(500, "Webhook secret not configured")

    payload = await request.body()

    # Verify signature
    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=stripe_signature,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except Exception as e:
        raise HTTPException(400, f"Invalid signature: {e}")

    event_type = event["type"]
    data_object = event["data"]["object"]

    print(f"📬 Stripe webhook received: {event_type}")

    # Helper to extract uid
    def get_uid(obj):
        return (obj.get("metadata") or {}).get("firebase_uid")

    # Subscription lifecycle events
    if event_type.startswith("customer.subscription."):
        sub = data_object
        firebase_uid = get_uid(sub)

        if not firebase_uid:
            print("⚠️ Missing firebase_uid in Stripe metadata")
            return {"received": True}

        status = sub.get("status", "canceled")

        # Get plan ID
        items = sub.get("items", {}).get("data", [])
        plan_id = items[0]["price"]["id"] if items else None

        upsert_subscription_for_uid(
            firebase_uid=firebase_uid,
            stripe_customer_id=sub.get("customer"),
            stripe_subscription_id=sub.get("id"),
            plan=plan_id,
            status=status,
            trial_end=sub.get("trial_end"),
            current_period_end=sub.get("current_period_end"),
            canceled_at=sub.get("canceled_at"),
        )

    # Checkout session → subscription mapping
    elif event_type == "checkout.session.completed":
        session = data_object
        firebase_uid = (session.get("metadata") or {}).get("firebase_uid")

        if not firebase_uid:
            print("⚠️ Missing firebase_uid in checkout.session")
            return {"received": True}

        subscription_id = session.get("subscription")
        customer_id = session.get("customer")

        if subscription_id:
            sub = stripe.Subscription.retrieve(subscription_id)
            status = sub.get("status", "canceled")
            items = sub.get("items", {}).get("data", [])
            plan_id = items[0]["price"]["id"] if items else None

            upsert_subscription_for_uid(
                firebase_uid=firebase_uid,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                plan=plan_id,
                status=status,
                trial_end=sub.get("trial_end"),
                current_period_end=sub.get("current_period_end"),
                canceled_at=sub.get("canceled_at"),
            )

    else:
        print(f"ℹ️ Ignoring unsupported event: {event_type}")

    return {"received": True}

# ─────────────────────────────────────────────
# Apple Store Notification (placeholder)
# ─────────────────────────────────────────────

@router.post("/apple-subscription-webhook")
async def apple_subscription_webhook(request: Request):
    body = await request.json()
    print("🍎 Received Apple notification keys:", list(body.keys()))
    return {"received": True}

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@router.get("/health-subscriptions")
async def health_subscriptions():
    return {"ok": True}
