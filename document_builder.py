import re
import ast
import pandas as pd
from config import COLUMN_MAP, CSV_SEP, DATA_PATH
from preprocess import clean_special_chars


def load_data(path=None):
    path = path or DATA_PATH
    df = pd.read_csv(path, sep=CSV_SEP)
    df = df.fillna("")
    return df


def get_col(row, key):
    col = COLUMN_MAP.get(key)
    if col and col in row.index:
        value = row[col]
        if pd.isna(value):
            return ""
        return str(value).strip()
    return ""


def format_int(value):
    if value == "":
        return ""
    try:
        f = float(value)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return value


MINUTE_RE = re.compile(r"(\d+)(?:\+(\d+))?")


def parse_minute(minute_str):
    if not minute_str:
        return None, False
    m = MINUTE_RE.search(minute_str)
    if not m:
        return None, False
    base = int(m.group(1))
    extra = int(m.group(2)) if m.group(2) else 0
    total = base + extra
    return total, total > 90


def parse_player_minute_list(value, penalty=False, own_goal=False):
    value = (value or "").strip()
    if not value:
        return []
    events = []
    for item in value.split("|"):
        item = item.strip()
        if not item:
            continue
        if "·" in item:
            name_part, minute_part = item.rsplit("·", 1)
        else:
            name_part, minute_part = item, ""
        is_pen = penalty or "(P)" in name_part
        name = name_part.replace("(P)", "").strip()
        minute, is_extra = parse_minute(minute_part.strip())
        events.append({
            "player": name,
            "minute": minute,
            "extra_time": is_extra,
            "penalty": is_pen,
            "own_goal": own_goal,
        })
    return events

def parse_long_field(value):
    value = (value or "").strip()
    if not value or value in ("[]", "nan"):
        return []

    value = clean_special_chars(value)

    try:
        items = ast.literal_eval(value)
        if not isinstance(items, list):
            return []
    except (ValueError, SyntaxError):
        inner = value.strip("[]")
        items = [s.strip(" '\"") for s in inner.split("', '") if s.strip(" '\"")]

    parsed = []
    for item in items:
        parts = [p.strip() for p in str(item).split("|")]
        parsed.append(parts)
    return parsed


def long_goal_events(value, penalty=False, own_goal=False):
    events = []
    for parts in parse_long_field(value):
        if len(parts) < 3:
            continue
        minute, is_extra = parse_minute(parts[0])
        player = parts[2]
        assist = ""
        if len(parts) >= 5 and parts[3].lower().startswith("assist"):
            assist = parts[4]
        events.append({
            "player": player, "minute": minute, "extra_time": is_extra,
            "penalty": penalty, "own_goal": own_goal, "assist": assist,
        })
    return events


def long_card_events(value):
    events = []
    for parts in parse_long_field(value):
        if len(parts) < 3:
            continue
        minute, is_extra = parse_minute(parts[0])
        player = parts[2]
        events.append({"player": player, "minute": minute, "extra_time": is_extra})
    return events


def long_substitution_events(events_str):
    if not events_str or events_str == "[]":
        return []

    events = []
    try:
        raw_list = ast.literal_eval(events_str)
    except (ValueError, SyntaxError):
        return []

    for item in raw_list:
        if not item:
            continue

        cleaned_item = clean_special_chars(str(item))
        parts = [p.strip() for p in cleaned_item.split("|")]

        if len(parts) >= 4:
            minute_str = parts[0]
            score_str = parts[1]
            player_in = parts[2]

            sub_info = parts[3]
            player_out = ""
            if sub_info.lower().startswith("for "):
                player_out = sub_info[4:].strip()
            else:
                player_out = sub_info

            minute, extra = parse_minute(minute_str)

            events.append({
                "type": "substitution",
                "minute": minute,
                "extra": extra,
                "score": score_str,
                "player_in": player_in,
                "player_out": player_out,
                "players": [player_in, player_out] if player_out else [player_in]
            })

    return events


def long_shootout_events(value):
    events = []
    for parts in parse_long_field(value):
        if not parts:
            continue
        player = parts[2] if len(parts) >= 3 else parts[-1]
        if player:
            events.append({"player": player})
    return events


def venue_parts(venue):
    venue = venue or ""
    if "," in venue:
        stadium, city = venue.split(",", 1)
        return stadium.strip(), city.strip()
    return venue.strip(), ""


def events_to_text(events, label, kind="goal"):
    pieces = []
    for e in events:
        player = e.get("player", "").strip()
        if not player:
            continue
        s = player
        minute = e.get("minute")
        if minute is not None:
            s += f" {minute}'"
        if kind == "goal":
            if e.get("penalty"):
                s += " penalty"
            if e.get("own_goal"):
                s += " own goal"
            if e.get("extra_time"):
                s += " extra time"
            if e.get("assist"):
                s += f" assist by {e['assist']}"
        elif kind == "card" and e.get("extra_time"):
            s += " extra time"
        pieces.append(s)
    if not pieces:
        return ""
    return f"{label}: " + ", ".join(pieces) + ". "


def substitutions_to_text(events, label):
    pieces = []
    for e in events:
        if not e.get("player_in"):
            continue
        s = e["player_in"]
        if e.get("minute") is not None:
            s += f" {e['minute']}'"
        if e.get("extra"):
            s += " extra time"
        if e.get("player_out"):
            s += f" for {e['player_out']}"
        pieces.append(s)
    if not pieces:
        return ""
    return f"{label}: " + ", ".join(pieces) + ". "


def shootout_to_text(scorers, missers, label):
    pieces = [f"{e['player']} scored" for e in scorers if e.get("player")]
    pieces += [f"{e['player']} missed" for e in missers if e.get("player")]
    if not pieces:
        return ""
    return f"{label}: " + ", ".join(pieces) + ". "


def red_card_text(events, label):
    if not events:
        return ""
    pieces = []
    for e in events:
        if not e.get("player"):
            continue
        s = e["player"]
        if e.get("minute") is not None:
            s += f" {e['minute']}'"
        pieces.append(s)
    if not pieces:
        return ""
    return f"{label}: " + ", ".join(pieces) + ". "


def build_document_text(row):
    home_team = get_col(row, "home_team")
    away_team = get_col(row, "away_team")
    round_ = get_col(row, "round")
    venue = get_col(row, "venue")
    stadium, city = venue_parts(venue)
    referee = get_col(row, "referee")
    notes = get_col(row, "notes")

    home_score = format_int(get_col(row, "home_score"))
    away_score = format_int(get_col(row, "away_score"))
    home_xg = get_col(row, "home_xg")
    away_xg = get_col(row, "away_xg")
    home_penalty = format_int(get_col(row, "home_penalty"))
    away_penalty = format_int(get_col(row, "away_penalty"))

    home_manager = get_col(row, "home_manager")
    away_manager = get_col(row, "away_manager")
    home_captain = get_col(row, "home_captain")
    away_captain = get_col(row, "away_captain")

    home_goal_long_val = get_col(row, "home_goal_long")
    if home_goal_long_val:
        base_home_goals = long_goal_events(home_goal_long_val)
    else:
        base_home_goals = parse_player_minute_list(get_col(row, "home_goal"))

    home_goals = (
            base_home_goals
            + parse_player_minute_list(get_col(row, "home_penalty_goal"), penalty=True)
            + parse_player_minute_list(get_col(row, "home_own_goal"), own_goal=True)
    )

    away_goal_long_val = get_col(row, "away_goal_long")
    if away_goal_long_val:
        base_away_goals = long_goal_events(away_goal_long_val)
    else:
        base_away_goals = parse_player_minute_list(get_col(row, "away_goal"))

    away_goals = (
            base_away_goals
            + parse_player_minute_list(get_col(row, "away_penalty_goal"), penalty=True)
            + parse_player_minute_list(get_col(row, "away_own_goal"), own_goal=True)
    )
    # -----------------------------------------------------

    home_pen_miss = long_goal_events(get_col(row, "home_penalty_miss_long"), penalty=True)
    away_pen_miss = long_goal_events(get_col(row, "away_penalty_miss_long"), penalty=True)

    home_yellow = long_card_events(get_col(row, "home_yellow_card_long"))
    away_yellow = long_card_events(get_col(row, "away_yellow_card_long"))

    home_subs = long_substitution_events(get_col(row, "home_substitute_in_long"))
    away_subs = long_substitution_events(get_col(row, "away_substitute_in_long"))

    home_shoot_goals = long_shootout_events(get_col(row, "home_penalty_shootout_goal_long"))
    home_shoot_miss = long_shootout_events(get_col(row, "home_penalty_shootout_miss_long"))
    away_shoot_goals = long_shootout_events(get_col(row, "away_penalty_shootout_goal_long"))
    away_shoot_miss = long_shootout_events(get_col(row, "away_penalty_shootout_miss_long"))

    home_red = parse_player_minute_list(get_col(row, "home_red_card"))
    away_red = parse_player_minute_list(get_col(row, "away_red_card"))
    home_yellow_red = parse_player_minute_list(get_col(row, "home_yellow_red_card"))
    away_yellow_red = parse_player_minute_list(get_col(row, "away_yellow_red_card"))

    parts = []
    parts.append(f"{home_team} vs {away_team}.")
    parts.append(f"Round: {round_}.")
    if stadium:
        parts.append(f"Stadium: {stadium}.")
    if city:
        parts.append(f"City: {city}.")

    if home_score != "" and away_score != "":
        parts.append(f"Score: {home_score}-{away_score}.")
        if home_score == away_score and (home_penalty == "" or away_penalty == ""):
            parts.append("Draw.")

    if home_xg or away_xg:
        parts.append(f"xG: {home_xg}-{away_xg}.")

    if referee:
        parts.append(f"Referee: {referee}.")
    if home_captain:
        parts.append(f"{home_team} captain: {home_captain}.")
    if away_captain:
        parts.append(f"{away_team} captain: {away_captain}.")
    if home_manager:
        parts.append(f"{home_team} coach: {home_manager}.")
    if away_manager:
        parts.append(f"{away_team} coach: {away_manager}.")

    parts.append(events_to_text(home_goals, f"{home_team} goals"))
    parts.append(events_to_text(away_goals, f"{away_team} goals"))
    parts.append(events_to_text(home_pen_miss, f"{home_team} penalty misses"))
    parts.append(events_to_text(away_pen_miss, f"{away_team} penalty misses"))
    parts.append(events_to_text(home_yellow, f"{home_team} yellow cards", kind="card"))
    parts.append(events_to_text(away_yellow, f"{away_team} yellow cards", kind="card"))
    parts.append(substitutions_to_text(home_subs, f"{home_team} substitutions"))
    parts.append(substitutions_to_text(away_subs, f"{away_team} substitutions"))

    parts.append(red_card_text(home_red, f"{home_team} red cards"))
    parts.append(red_card_text(away_red, f"{away_team} red cards"))
    parts.append(red_card_text(home_yellow_red, f"{home_team} second yellow card red cards"))
    parts.append(red_card_text(away_yellow_red, f"{away_team} second yellow card red cards"))

    if home_penalty != "" and away_penalty != "":
        try:
            winner = home_team if float(home_penalty) > float(away_penalty) else away_team
        except ValueError:
            winner = ""
        parts.append(
            f"Penalty shootout: {winner} won {home_penalty}-{away_penalty}. "
            "Penalty shootout decided the match."
        )
        parts.append(shootout_to_text(home_shoot_goals, home_shoot_miss, f"{home_team} penalty shootout"))
        parts.append(shootout_to_text(away_shoot_goals, away_shoot_miss, f"{away_team} penalty shootout"))

    if notes:
        parts.append(f"Notes: {notes}.")

    all_goals = home_goals + away_goals
    if any(g.get("extra_time") for g in all_goals):
        parts.append("Extra time goal.")
    if any(g.get("penalty") for g in all_goals):
        parts.append("Penalty goal.")
    if any(g.get("own_goal") for g in all_goals):
        parts.append("Own goal.")

    return " ".join(p for p in parts if p)


def build_documents(df):
    documents = {}
    for idx, row in df.iterrows():
        doc_id = int(idx)
        text = build_document_text(row)

        home_team = get_col(row, "home_team")
        away_team = get_col(row, "away_team")
        round_ = get_col(row, "round")
        venue = get_col(row, "venue")
        stadium, city = venue_parts(venue)
        referee = get_col(row, "referee")
        home_score = format_int(get_col(row, "home_score"))
        away_score = format_int(get_col(row, "away_score"))
        attendance = get_col(row, "attendance")

        players = set()

        for key in ("home_goal_long", "away_goal_long"):
            for e in long_goal_events(get_col(row, key)):
                if e.get("player"):
                    players.add(e["player"])
                if e.get("assist"):
                    players.add(e["assist"])

        if not get_col(row, "home_goal_long"):
            for e in parse_player_minute_list(get_col(row, "home_goal")):
                if e.get("player"):
                    players.add(e["player"])

        if not get_col(row, "away_goal_long"):
            for e in parse_player_minute_list(get_col(row, "away_goal")):
                if e.get("player"):
                    players.add(e["player"])

        for key, kwargs in [
            ("home_penalty_goal", {"penalty": True}),
            ("away_penalty_goal", {"penalty": True}),
            ("home_own_goal", {"own_goal": True}),
            ("away_own_goal", {"own_goal": True}),
        ]:
            for e in parse_player_minute_list(get_col(row, key), **kwargs):
                if e.get("player"):
                    players.add(e["player"])

        for key in ("home_red_card", "away_red_card", "home_yellow_red_card", "away_yellow_red_card"):
            for e in parse_player_minute_list(get_col(row, key)):
                if e.get("player"):
                    players.add(e["player"])

        for key in ("home_yellow_card_long", "away_yellow_card_long"):
            for e in long_card_events(get_col(row, key)):
                if e.get("player"):
                    players.add(e["player"])

        for key in ("home_penalty_miss_long", "away_penalty_miss_long"):
            for e in long_goal_events(get_col(row, key)):
                if e.get("player"):
                    players.add(e["player"])
                if e.get("assist"):
                    players.add(e["assist"])

        for key in ("home_substitute_in_long", "away_substitute_in_long"):
            for e in long_substitution_events(get_col(row, key)):
                if e.get("player_in"):
                    players.add(e["player_in"])
                if e.get("player_out"):
                    players.add(e["player_out"])

        for key in (
            "home_penalty_shootout_goal_long", "away_penalty_shootout_goal_long",
            "home_penalty_shootout_miss_long", "away_penalty_shootout_miss_long",
        ):
            for e in long_shootout_events(get_col(row, key)):
                if e.get("player"):
                    players.add(e["player"])

        fields = {
            "team": f"{home_team} {away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "round": round_,
            "stage": round_,
            "stadium": stadium,
            "city": city,
            "venue": venue,
            "referee": referee,
            "captain": f"{get_col(row, 'home_captain')} {get_col(row, 'away_captain')}",
            "coach": f"{get_col(row, 'home_manager')} {get_col(row, 'away_manager')}",
            "player": " ".join(sorted(players)),
            "score": f"{home_score}-{away_score}",
            "year": format_int(get_col(row, "year")),
            "host": get_col(row, "host"),
            "attendance": attendance,
        }

        metadata = {
            "home_team": home_team,
            "away_team": away_team,
            "round": round_,
            "stadium": stadium,
            "city": city,
            "score": f"{home_score}-{away_score}",
            "referee": referee,
            "date": get_col(row, "date"),
            "attendance": attendance,
            "year": format_int(get_col(row, "year")),
        }

        documents[doc_id] = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata,
            "fields": fields,
        }
    return documents