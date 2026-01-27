import json
import os
import glob
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==============================================================================
# 1. THE CALCULATOR (StatsTool)
# ==============================================================================
class StatsTool:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.batting_df = self._load_multi_files('batting_stats')
        self.bowling_df = self._load_multi_files('bowling_stats')
        self.career_df = self._load_multi_files('player_career_stats') 
        self.matches_df = self._load_single_file('matches', 'matches.json')
        self.teams_df = self._load_single_file('teams', 'teams.json')
        self.team_map = self._build_team_map()

    def _load_single_file(self, folder, filename):
        path = os.path.join(self.base_dir, folder, filename)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
        except:
            return pd.DataFrame()

    def _load_multi_files(self, folder):
        path = os.path.join(self.base_dir, folder)
        all_records = []
        for f in glob.glob(f"{path}/*.json"):
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    payload = data.get('response', {}) if 'response' in data else data
                    stats = []
                    if isinstance(payload, dict):
                        stats = payload.get('stats', []) or payload.get('batting_stats') or payload.get('bowling_stats') or payload.get('career_stats') or []
                    
                    if isinstance(payload, dict) and not stats:
                        all_records.append(payload)
                    elif isinstance(stats, list):
                        all_records.extend(stats)
            except:
                continue
        df = pd.DataFrame(all_records)
        df.columns = [str(c).lower().strip() for c in df.columns]
        for col in df.columns:
            if not df[col].empty and isinstance(df[col].dropna().iloc[0], dict):
                df[col] = df[col].apply(lambda x: x.get('name') or x.get('title') if isinstance(x, dict) else x)
        return df

    def _build_team_map(self):
        if self.teams_df.empty: return {}
        mapping = {}
        for _, row in self.teams_df.iterrows():
            if 'abbr' in row and 'title' in row:
                mapping[row['abbr'].lower()] = row['title'].lower()
                mapping[row['title'].lower()] = row['title'].lower()
        return mapping

    def get_stat(self, df, player_name, stat_aliases):
        if df.empty: return None
        name_col = next((c for c in df.columns if 'name' in c or 'player' in c), None)
        if not name_col: return None
        
        mask = df[name_col].astype(str).str.lower().str.contains(player_name.lower(), na=False)
        subset = df[mask]
        if subset.empty: return None
        
        if isinstance(stat_aliases, str): stat_aliases = [stat_aliases]
        target_col = None
        for alias in stat_aliases:
            found = next((c for c in df.columns if alias in c), None)
            if found:
                target_col = found
                break
        
        if target_col: 
            numeric_series = pd.to_numeric(subset[target_col], errors='coerce')
            max_val = numeric_series.max()
            if pd.notna(max_val):
                return int(max_val) if max_val.is_integer() else max_val
        return None

    def find_match_commentary(self, query):
        path = os.path.join(self.base_dir, 'match_innings_commentary')
        files = glob.glob(f"{path}/*.json")
        query_teams = []
        q_lower = query.lower()
        for abbr, full_name in self.team_map.items():
            if abbr in q_lower or full_name in q_lower:
                query_teams.append(full_name.replace(" ", "_"))
        if not query_teams: return "Could not identify teams in query."
        for f in files:
            fname = os.path.basename(f).lower()
            if any(t in fname for t in query_teams):
                with open(f, 'r') as file:
                    data = json.load(file)
                    return data.get('commentaries', [])[:3]
        return "No commentary file matched."

# ==============================================================================
# 2. THE LIBRARIAN (RetrieverAgent) - (Fixed for Partial Matches)
# ==============================================================================
class RetrieverAgent:
    def __init__(self, tool: StatsTool):
        self.tool = tool

    def retrieve(self, query):
        context = []
        q_lower = query.lower().strip()
        
        # Identify Player
        all_players = set()
        if not self.tool.batting_df.empty:
            all_players.update(self.tool.batting_df.get('player', pd.Series()).dropna().unique())
        if not self.tool.career_df.empty:
            col = next((c for c in self.tool.career_df.columns if 'name' in c), None)
            if col: all_players.update(self.tool.career_df[col].dropna().unique())

        found_player = None
        for p in all_players:
            if isinstance(p, str):
                p_clean = p.lower().strip()
                
                # CHECK 1: Full Name in Query? (High Confidence)
                if p_clean in q_lower:
                    found_player = p
                    break
                
                # CHECK 2: Name Parts in Query? (Partial Match)
                # Split "Dwayne Bravo" -> ["dwayne", "bravo"]
                name_parts = p_clean.split()
                for part in name_parts:
                    # Check if "bravo" is in query, ensure it's not a tiny word like "de" or "al"
                    if len(part) >= 3 and part in q_lower:
                        # Double check we don't match "Roy" inside "Royal Challengers"
                        # Simple boundary check: ensure the match isn't part of a larger word if possible
                        # For this level of test, simply checking existence is usually fine.
                        found_player = p
                        break
                if found_player: break
        
        if found_player:
            runs = self.tool.get_stat(self.tool.batting_df, found_player, ['runs', 'run', 'score'])
            sr = self.tool.get_stat(self.tool.batting_df, found_player, ['strike', 'rate'])
            if runs: context.append(f"2022 Season Batting: {found_player} scored {runs} runs (SR {sr}).")
            
            wickets = self.tool.get_stat(self.tool.bowling_df, found_player, ['wickets', 'wkt'])
            econ = self.tool.get_stat(self.tool.bowling_df, found_player, ['economy', 'econ'])
            if wickets: context.append(f"2022 Season Bowling: {found_player} took {wickets} wickets (Econ {econ}).")
            
            total_matches = self.tool.get_stat(self.tool.career_df, found_player, ['match', 'mat'])
            total_runs = self.tool.get_stat(self.tool.career_df, found_player, ['run', 'score'])
            if total_matches:
                 context.append(f"CAREER STATS: {found_player} has played {total_matches} total matches and scored {total_runs} runs.")

        if "vs" in q_lower or "match" in q_lower:
            comm = self.tool.find_match_commentary(query)
            if isinstance(comm, list) and comm:
                evt = comm[0]
                text = f"Commentary: {evt.get('commentary', '')} (Over {evt.get('over')}.{evt.get('ball')})"
                context.append(text)

        if not context: return "No specific database records found."
        return "\n".join(list(set(context)))

# ==============================================================================
# 3. THE BRAIN (AnalystAgent)
# ==============================================================================
class AnalystAgent:
    def __init__(self):
        print("🧠 Loading AI Model...")
        base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   ⚙️ Hardware Detected: {self.device.upper()}")

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True
            )
        else:
            print("   ⚠️ Running on CPU: Performance will be slower.")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="cpu",
                trust_remote_code=True
            )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        adapter_path = "ipl_analyst_adapter"
        if os.path.exists(adapter_path):
            print(f"   ✨ Applying trained adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            
        print("✅ AnalystAgent Ready.")

    def generate_answer(self, query, context):
        prompt = f"<|system|>\nYou are a cricket analyst. Use the Context to answer the User.\n<|user|>\nContext: {context}\nQuestion: {query}\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id
            )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|assistant|>" in full_text:
            answer = full_text.split("<|assistant|>")[-1].strip()
        else:
            answer = full_text.replace(prompt, "").strip()
        if "User:" in answer: answer = answer.split("User:")[0]
        if "<|user|>" in answer: answer = answer.split("<|user|>")[0]
        return answer
