import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal, Optional, List
from dataclasses import dataclass, asdict

print("\n" + "💼" * 50)
print("🚀 AI SDR AGENT - DAY 5 TUTORIAL")

print("💡 agent.py LOADED SUCCESSFULLY!")
print("💼" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")



FAQ_FILE = "store_faq.json"
LEADS_FILE = "leads_db.json"

DEFAULT_FAQ = [
  {
    "question": "What does Razorpay do?",
    "answer": "Razorpay provides online payment solutions that allow businesses to accept payments via cards, UPI, net banking, and wallets, along with tools for invoicing, subscriptions, and business banking."
  },
  {
    "question": "Who is this for?",
    "answer": "Razorpay is built for startups, ecommerce businesses, SaaS platforms, freelancers, marketplaces, and enterprises that require secure and scalable payment infrastructure."
  },
  {
    "question": "Do you have a free tier?",
    "answer": "Razorpay has no setup or annual fees. Businesses only pay a transaction fee when they successfully receive payments."
  },
  {
    "question": "What are your pricing details?",
    "answer": "Standard payment gateway pricing starts at approximately 2% per transaction depending on the payment mode. Additional services may have separate usage-based charges."
  },
  {
    "question": "Can I use Razorpay for subscriptions?",
    "answer": "Yes, Razorpay supports recurring billing and subscription management for SaaS and membership-based businesses."
  },
  {
    "question": "Does Razorpay support international payments?",
    "answer": "Yes, international payments are supported with additional compliance requirements and activation steps."
  },
  {
    "question": "Is Razorpay secure?",
    "answer": "Razorpay follows industry-grade security standards including PCI DSS compliance and encrypted transactions."
  }
]


def load_knowledge_base():
    """Generates FAQ file if missing, then loads it."""
    try:
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            return json.dumps(json.load(f)) # Return as string for the Prompt
    except Exception as e:
        print(f"⚠️ Error loading FAQ: {e}")
        return ""

STORE_FAQ_TEXT = load_knowledge_base()



@dataclass
class LeadProfile:
    name: str | None = None
    company: str | None = None
    email: str | None = None
    role: str | None = None
    use_case: str | None = None
    team_size: str | None = None
    timeline: str | None = None
   
    def is_qualified(self):
        """Returns True if we have the minimum info (Name + Email + Use Case)"""
        return all([self.name, self.email, self.use_case])

@dataclass
class Userdata:
    lead_profile: LeadProfile



@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's company name")] = None,
    email: Annotated[Optional[str], Field(description="Customer's email address")] = None,
    role: Annotated[Optional[str], Field(description="Customer's job title")] = None,
    use_case: Annotated[Optional[str], Field(description="What they want to build or learn")] = None,
    team_size: Annotated[Optional[str], Field(description="Number of people in their team")] = None,
    timeline: Annotated[Optional[str], Field(description="When they want to start (e.g., Now, next month)")] = None,
) -> str:
    """
    ✍️ Captures lead details provided by the user during conversation.
    Only call this when the user explicitly provides information.
    """
    profile = ctx.userdata.lead_profile
   
    # Update only fields that are provided (not None)
    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline
   
    print(f"📝 UPDATING LEAD: {profile}")
    return "Lead profile updated. Continue the conversation."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
) -> str:
    """
    💾 Saves the lead to the database and signals the end of the call.
    Call this when the user says goodbye or 'that's all'.
    """
    profile = ctx.userdata.lead_profile
   
    # Save to JSON file (Append mode)
    db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)
   
    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()
   
    # Read existing, append, write back (Simple JSON DB)
    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                existing_data = json.load(f)
        except: pass
   
    existing_data.append(entry)
   
    with open(db_path, "w") as f:
        json.dump(existing_data, f, indent=4)
       
    print(f"✅ LEAD SAVED TO {LEADS_FILE}")
    return f"Lead saved. Summarize the call for the user: 'Thanks {profile.name}, I have your info regarding {profile.use_case}. We will email you at {profile.email}. Goodbye!'"



class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are 'Riya', a sharp, professional Sales Development Representative (SDR) for 'Razorpay'.

            📘 YOUR KNOWLEDGE BASE (FAQ):
            {DEFAULT_FAQ}

            🎯 YOUR GOAL:
            1. Answer questions about Razorpay's products, pricing, and services using ONLY the provided FAQ.
            2. QUALIFY THE LEAD by naturally collecting:
               - Name
               - Company
               - Role
               - Email
               - Use Case (what they want to build or enable)
               - Team Size
               - Timeline (Now / Soon / Later)

            ⚙️ BEHAVIOR:
            - Be conversational, not robotic.
            - First help. Then probe.
              Example: "Razorpay supports subscription billing. By the way, may I know what you're planning to build?"
            - Capture data immediately using `update_lead_profile` when new info is revealed.
            - Detect end-of-conversation phrases like "That's all", "Thanks", "I'm done".
            - On closure, call `submit_lead_and_end` with summary.

            🚫 RESTRICTIONS:
            - DO NOT invent features or pricing.
            - If info isn't in FAQ, say: 
              "I don’t want to give you incorrect info — our team will clarify this for you."

            ✅ OPENING LINE:
            "Hi! Welcome to Razorpay. I'm Riya. What brings you here today and what are you working on?"
            """,
            tools=[update_lead_profile, submit_lead_and_end],
        )




def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "💼" * 25)
    print("🚀 STARTING SDR SESSION")
   
    # 1. Initialize State
    userdata = Userdata(lead_profile=LeadProfile())

    # 2. Setup Agent
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie", # Professional, warm female voice
            style="Promo",        
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
   
    # 3. Start
    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
