import os
import sys

# Ensure Python looks in the current folder for modules
sys.path.append(os.getcwd())

from agents.multi_agent import StatsTool, RetrieverAgent, AnalystAgent

# ==========================================
# 1. CONFIGURATION
# ==========================================
# This assumes your data is in: IPL_Project/data/Indian_Premier_League...
# If your folder name is different, update it here.
DATA_PATH = os.path.join(
    os.getcwd(), 
    'data', 
    'Indian_Premier_League_2022-03-26', 
    'Indian_Premier_League_2022-03-26'
)

# ==========================================
# 2. INITIALIZATION
# ==========================================
print(f"📂 Checking Data Path: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print("\n❌ ERROR: Data folder not found!")
    print("   Make sure you created a 'data' folder and copied the files inside.")
    print("   Your structure should be: IPL_Project -> data -> Indian_Premier_League...")
    sys.exit(1)

print("⚙️  Initializing StatsTool...")
tool = StatsTool(DATA_PATH)
retriever = RetrieverAgent(tool)

print("🧠 Loading AI Model...")
analyst = AnalystAgent()

# ==========================================
# 3. CHAT LOOP
# ==========================================
print("\n" + "="*50)
print("💬 IPL ANALYST CHAT (Type 'exit' to stop)")
print("="*50)

while True:
    try:
        # Get User Input
        query = input("\n❓ You: ").strip()
        
        # Check for exit command
        if query.lower() in ['exit', 'quit', 'stop']:
            print("👋 Exiting chat. Goodbye!")
            break
        
        if not query: continue

        # Step A: Retrieve
        print("   🔎 Searching database...")
        context = retriever.retrieve(query)
        
        # Step B: Generate
        print("   🤖 Thinking...")
        answer = analyst.generate_answer(query, context)
        
        # Step C: Output
        print(f"💡 AI: {answer}")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\n\n👋 Exiting chat. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
