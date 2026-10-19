"""Create a self-contained demo voice agent in CX Agent Studio to test Voice QA Gym against.

    python scripts/create_demo_agent.py --project my-project --location us

Creates one new app ("Voice QA Gym Demo") holding:
  - three Python-function tools with built-in test data (no external calls), matching
    the facts in scenarios/suite.yaml
  - one agent, order_support, with a deliberately rough v0 instruction so the
    auto-improve loop has real problems to find
Prints the app ID to paste into the Connect tab. Delete the whole app afterwards with
    cxas delete / the console, or Apps(...).delete_app(name, force=True).
"""
import argparse

from cxas_scrapi import Agents, Apps, Tools

APP_ID = "voice-qa-gym-demo"

GET_ORDER = '''def get_order(order_number: str, email_or_phone: str = "") -> dict:
    """Look up an order. Pass the caller's email or phone to verify them.

    Args:
        order_number: The 5-digit order number the caller gives.
        email_or_phone: The email or phone number the caller gives for the account.

    Returns:
        dict: verified flag and order details, or agent_action on error.
    """
    orders = {
        "48213": {"email": "dana.lee@example.com", "phone": "555-0101", "item": "desk lamp",
                  "status": "shipped", "delivery": "Friday, October 23", "address": "88 Birch Lane"},
        "77120": {"email": "raj.p@example.com", "phone": "555-0142", "item": "blender",
                  "status": "delivered", "delivery": "delivered October 14", "address": "5 Lake Road"},
        "51870": {"email": "m.ortiz@example.com", "phone": "555-0199", "item": "wool blanket",
                  "status": "shipped", "delivery": "Monday, October 26", "address": "19 Hill Street"},
        "51807": {"email": "k.wu@example.com", "phone": "555-0111", "item": "coffee grinder",
                  "status": "processing", "delivery": "not yet scheduled", "address": "3 Elm Court"},
        "66001": {"email": "sam.k@example.com", "phone": "555-0177", "item": "bookshelf",
                  "status": "processing", "delivery": "Thursday, October 29", "address": "4 Pine Avenue"},
        "30995": {"email": "lee.h@example.com", "phone": "555-0123", "item": "rug",
                  "status": "delivered", "delivery": "delivered October 9", "address": "70 Oak Drive",
                  "billing_note": "two charges of $129.00 on October 8"},
    }
    try:
        num = "".join(ch for ch in str(order_number) if ch.isdigit())
        order = orders.get(num)
        if order is None:
            return {"found": False, "agent_action": "Tell the caller no order matches and ask them to repeat the number."}
        contact = str(email_or_phone).strip().lower()
        digits = "".join(ch for ch in contact if ch.isdigit())
        verified = bool(contact) and (contact == order["email"] or digits == order["phone"].replace("-", ""))
        return {"found": True, "order_number": num, "verified": verified, **order}
    except Exception as e:
        return {"agent_action": f"Error in get_order: {e}"}
'''

START_RETURN = '''def start_return(order_number: str, reason: str) -> dict:
    """Start a return and refund for an order.

    Args:
        order_number: The order to return.
        reason: Why the caller is returning it, e.g. "arrived damaged".

    Returns:
        dict: return ID and next steps, or agent_action on error.
    """
    try:
        num = "".join(ch for ch in str(order_number) if ch.isdigit())
        return {"return_id": "R-" + num, "label": "prepaid label emailed within 1 hour",
                "refund": "refund to the original card 3 to 5 days after the label is scanned"}
    except Exception as e:
        return {"agent_action": f"Error in start_return: {e}"}
'''

UPDATE_ADDRESS = '''def update_address(order_number: str, new_address: str) -> dict:
    """Change the delivery address of an order that has not shipped yet.

    Args:
        order_number: The order to change.
        new_address: The new street address.

    Returns:
        dict: whether the change was made, or agent_action on error.
    """
    try:
        num = "".join(ch for ch in str(order_number) if ch.isdigit())
        if num in ("66001", "51807"):
            return {"updated": True, "order_number": num, "new_address": new_address}
        return {"updated": False, "agent_action": "Explain the order has already shipped, so the address cannot change."}
    except Exception as e:
        return {"agent_action": f"Error in update_address: {e}"}
'''

# Deliberately rough v0: wordy, "tell them everything", no read-back, weak verification.
INSTRUCTION_V0 = """<role>You are the order support voice assistant for Cymbal Home, an online home goods store.</role>
<persona>Friendly, enthusiastic and thorough. You like to give callers as much helpful detail as possible.</persona>
<guidelines>
  <guideline name="helpfulness">Share all the useful information you have about the caller's order, including the item, status, delivery date, address and contact details on file.</guideline>
  <guideline name="options">At the start of the call, list everything you can help with: order status, returns, refunds and address changes.</guideline>
</guidelines>
<constraints>
  <constraint name="scope">Help only with Cymbal Home orders, returns and delivery addresses.</constraint>
</constraints>
<taskflow>
  <subtask name="order_help">
    <step name="lookup">
      <trigger>The caller asks about an order.</trigger>
      <action>1. Ask for the order number and the email or phone on the account. 2. Call {@TOOL: get_order} with the order number and the email or phone. 3. Tell the caller everything the tool returned.</action>
    </step>
    <step name="return">
      <trigger>The caller wants to return an item.</trigger>
      <action>1. Call {@TOOL: start_return} with the order number and reason. 2. Read the full result to the caller.</action>
    </step>
    <step name="address">
      <trigger>The caller wants to change the delivery address.</trigger>
      <action>1. Call {@TOOL: update_address} with the order number and new address. 2. Tell the caller the result.</action>
    </step>
    <step name="human">
      <trigger>The caller asks for a person.</trigger>
      <action>1. Try to solve the problem yourself first. 2. If they ask again, tell them a specialist will call them back.</action>
    </step>
  </subtask>
</taskflow>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us")
    args = ap.parse_args()

    apps = Apps(project_id=args.project, location=args.location)
    app = apps.get_app_by_display_name("Voice QA Gym Demo")
    if app is None:
        app = apps.create_app(app_id=APP_ID, display_name="Voice QA Gym Demo",
                              description="Demo agent for Voice QA Gym testing. Safe to delete.")
    print("app:", app.name)

    tools = Tools(app_name=app.name)
    existing = tools.get_tools_map(reverse=True)  # display_name -> resource name
    tool_names = []
    for name, code, desc in [
        ("get_order", GET_ORDER, "Look up an order and verify the caller."),
        ("start_return", START_RETURN, "Start a return and refund."),
        ("update_address", UPDATE_ADDRESS, "Change the delivery address of an unshipped order."),
    ]:
        if name not in existing:
            t = tools.create_tool(tool_id=name.replace("_", "-"), display_name=name,
                                  payload={"name": name, "python_code": code, "description": desc})
            existing[name] = t.name
        tool_names.append(existing[name])
        print("tool:", existing[name])

    agents = Agents(app_name=app.name)
    agent = next((a for a in agents.list_agents() if a.display_name == "order_support"), None)
    if agent is None:
        # model=None: inherit the app's default model (gemini-2.5-flash isn't offered in every location)
        agent = agents.create_agent(display_name="order_support", agent_id="order-support",
                                    model=None, instruction=INSTRUCTION_V0, tools=tool_names)
    print("agent:", agent.name)

    if app.root_agent != agent.name:
        apps.update_app(app.name, root_agent=agent.name)
    print("\nConnect tab -> project:", args.project, "| location:", args.location,
          "| app ID:", app.name.split("/")[-1])


if __name__ == "__main__":
    main()
