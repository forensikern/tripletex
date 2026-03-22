"""
Comprehensive test runner — covers all 7 task categories with varied prompts.

Usage:
    python test_local.py              # all tasks
    python test_local.py 5            # by index
    python test_local.py emp          # by keyword

Set credentials:
    $env:SANDBOX_TOKEN="eyJ..."
    $env:ANTHROPIC_API_KEY="sk-ant-..."
    $env:LLM_PROVIDER="anthropic"
    $env:LLM_MODEL="claude-haiku-4-5-20251001"   # cheap for testing
    $env:LLM_MODEL="claude-opus-4-6"             # best for competition
"""

import os, sys, json, time, logging
from datetime import datetime
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(__file__))
from main import (build_user_content, build_user_content_anthropic,
                  process_tool_call, PROVIDER, MODEL, SYSTEM_PROMPT,
                  TOOLS, ANTHROPIC_TOOLS, OAI_CLIENT, ANTHROPIC_CLIENT)

BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
TOKEN    = os.environ.get("SANDBOX_TOKEN", "YOUR_SESSION_TOKEN_HERE")

# Unique suffix per run to avoid email/name collisions across test runs
RUN_ID = datetime.now().strftime("%H%M%S")

def u(s):
    """Make a string unique per run — avoids 'email already in use' errors."""
    return s.replace("@", f".{RUN_ID}@")

def n(name, suffix=""):
    """Make a company/person name unique per run."""
    return f"{name} {RUN_ID}{suffix}"


def make_tasks():
    return [
        # ═══ EMPLOYEES ═══════════════════════════════════════════════
        {
            "name": "EMP-01: Opprett administrator",
            "prompt": f"Opprett en ansatt med navn Erik Solberg, e-post {u('erik.solberg@example.com')}. Han skal ha kontoadministratortilgang.",
        },
        {
            "name": "EMP-02: Opprett vanlig ansatt",
            "prompt": f"Opprett en ny ansatt: fornavn Ingrid, etternavn Dahl, e-post {u('ingrid.dahl@example.com')}. Standard tilgang.",
        },
        {
            "name": "EMP-03: Oppdater arbeidstelefon",
            "prompt": "Finn ansatt med e-post olssonc@gmail.com og sett arbeidstelefon til 23456789.",
        },
        {
            "name": "EMP-04: Oppdater mobiltelefon",
            "prompt": "Finn den første ansatte i systemet og oppdater mobiltelefonnummeret til 98765432.",
        },
        {
            "name": "EMP-05: Opprett ansatt med kommentar",
            "prompt": f"Opprett ansatt Tor Hansen, e-post {u('tor.hansen@example.com')}. Legg til kommentaren 'Innleid konsulent'.",
        },

        # ═══ CUSTOMERS & PRODUCTS ═══════════════════════════════════
        {
            "name": "CUS-01: Opprett kunde med org.nr",
            "prompt": f"Registrer en ny kunde: {n('Havfisk Gruppen AS')}, epost: {u('post@havfisk.no')}, organisasjonsnummer 998877665.",
        },
        {
            "name": "CUS-02: Opprett leverandør",
            "prompt": f"Opprett en leverandør med navn {n('Kontorrekvisita AS')}, e-post {u('ordre@kontor.no')}.",
        },
        {
            "name": "CUS-03: Oppdater e-post på kunde",
            "prompt": f"Finn kunden med navn '{n('Havfisk Gruppen AS')}' og oppdater e-postadressen til {u('faktura@havfisk.no')}.",
        },
        {
            "name": "CUS-04: Oppdater telefon på kunde",
            "prompt": "Finn kunden Bergvik AS og oppdater telefonnummeret til 55667788.",
        },
        {
            "name": "PRD-01: Opprett produkt",
            "prompt": f"Opprett et nytt produkt med navn 'Systemintegrasjon {RUN_ID}', produktnummer 'SYS-{RUN_ID}', pris 4500 kr ekskl. mva.",
        },
        {
            "name": "PRD-02: Opprett og oppdater produktpris",
            "prompt": f"Opprett produktet 'Månedlig support {RUN_ID}', nummer 'SUP-{RUN_ID}', pris 1200 kr. Oppdater deretter prisen til 1350 kr.",
        },
        {
            "name": "PRD-03: Oppdater eksisterende pris",
            "prompt": "Finn produktet med navn 'Konsulenttjeneste' og oppdater prisen til 1800 kr ekskl. mva.",
        },

        # ═══ INVOICING ═══════════════════════════════════════════════
        {
            "name": "INV-01: Opprett faktura",
            "prompt": f"Opprett en kunde {n('Bryggen Handel AS')} (epost: {u('bh@test.no')}). Lag en faktura med én linje: 'Prosjektledelse', antall 5, pris 2000 kr/stk eks. mva. Fakturadato i dag.",
        },
        {
            "name": "INV-02: Faktura med to linjer",
            "prompt": f"Opprett kunde {n('Dobbel Linje AS')} (epost: {u('dl@test.no')}). Faktura med to linjer: 1) 'Utvikling' antall 10, pris 1200 kr. 2) 'Design' antall 4, pris 900 kr. Fakturadato i dag.",
        },
        {
            "name": "INV-03: Faktura + betaling",
            "prompt": f"Ny kunde: {n('Kontant Betaler AS')} (epost: {u('kb@test.no')}). Faktura: 'Konsultasjon', antall 2, pris 3500 kr. Fakturadato i dag. Registrer full betaling.",
        },
        {
            "name": "INV-04: Faktura + kreditnota",
            "prompt": f"Ny kunde: {n('Returvare AS')} (epost: {u('rv@test.no')}). Faktura: 'Vare', antall 1, pris 8000 kr. Fakturadato i dag. Utsted kreditnota på denne fakturaen.",
        },
        {
            "name": "INV-05: Finn nyeste faktura og betal",
            "prompt": "Finn den nyeste fakturaen i systemet og registrer betaling i sin helhet. Bruk dagens dato.",
        },

        # ═══ TRAVEL EXPENSES ═════════════════════════════════════════
        {
            "name": "TRV-01: Registrer reiseregning",
            "prompt": "Registrer en reiseregning for første ansatt i systemet. Tittel: 'Besøk kunde i Stavanger', dato i dag.",
        },
        {
            "name": "TRV-02: Opprett og slett reiseregning",
            "prompt": f"Opprett en reiseregning 'Kurs Oslo {RUN_ID}' for første ansatt, dato i dag. Slett den deretter.",
        },
        {
            "name": "TRV-03: Slett nyeste reiseregning",
            "prompt": "Finn den nyeste reiseregningen i systemet og slett den.",
        },
        {
            "name": "TRV-04: Reiseregning for spesifikk ansatt",
            "prompt": "Opprett en reiseregning med tittel 'Fagdag Trondheim' for ansatt med e-post olssonc@gmail.com. Dato: i dag.",
        },
        {
            "name": "TRV-05: Reiseregning med kostnad",
            "prompt": f"Opprett en reiseregning 'Fagmesse Berlin {RUN_ID}' for første ansatt, dato i dag. Legg til en kostnad på 2500 kr.",
        },
        {
            "name": "TRV-06: Reiseregning med flere kostnader",
            "prompt": f"Opprett en reiseregning 'Konferanse Stockholm {RUN_ID}' for første ansatt, dato i dag. Legg til to kostnader: flybillett 3200 kr og hotell 1800 kr.",
        },
        {
            "name": "TRV-07: Reiseregning med diett (hotell)",
            "prompt": f"Registrer ei reiseregning for første ansatt. Tittel: 'Kundebesøk Bergen {RUN_ID}'. Reisa varte 5 dagar med diett og hotellovernatting.",
        },
        {
            "name": "TRV-08: Reiseregning med diett (ingen overnatting)",
            "prompt": f"Opprett reiseregning 'Dagskurs {RUN_ID}' for første ansatt, dato i dag. 2 dager med diett, ingen overnatting.",
        },
        {
            "name": "TRV-09: Reiseregning med kjøregodtgjørelse",
            "prompt": f"Opprett en reiseregning 'Kundemøte {RUN_ID}' for første ansatt, dato i dag. Legg til kjøregodtgjørelse: 120 km fra Oslo til Drammen.",
        },
        {
            "name": "TRV-10: Reiseregning med diett + kjøregodtgjørelse",
            "prompt": f"Registrer reiseregning 'Fagtur Bergen {RUN_ID}' for første ansatt. 3 dager, hotellovernatting. Legg til diett og 450 km kjøregodtgjørelse (Oslo til Bergen).",
        },

        # ═══ PROJECTS ════════════════════════════════════════════════
        {
            "name": "PRJ-01: Internt prosjekt",
            "prompt": f"Opprett et internt prosjekt med navn 'Interne Systemer {RUN_ID}', prosjektnummer 'INT-{RUN_ID}', startdato 2026-05-01.",
        },
        {
            "name": "PRJ-02: Prosjekt knyttet til ny kunde",
            "prompt": f"Opprett en kunde {n('Byggmester AS')} (epost: {u('bygg@test.no')}). Opprett prosjekt 'Nybygg Vest {RUN_ID}', nummer 'BV-{RUN_ID}', startdato i dag, koblet til denne kunden.",
        },
        {
            "name": "PRJ-03: Finn og slett prosjekt",
            "prompt": f"Finn prosjektet med navn 'Interne Systemer {RUN_ID}' og slett det.",
        },
        {
            "name": "PRJ-04: Prosjekt med sluttdato",
            "prompt": f"Opprett prosjekt 'Q2 Leveranse {RUN_ID}', nummer 'Q2-{RUN_ID}', startdato 2026-04-01, sluttdato 2026-06-30. Ikke internt.",
        },

        # ═══ DEPARTMENTS ═════════════════════════════════════════════
        {
            "name": "DEP-01: Opprett avdeling",
            "prompt": f"Opprett en ny avdeling med navn 'Teknologi {RUN_ID}' og avdelingsnummer 'T{RUN_ID}'.",
        },
        {
            "name": "DEP-02: Opprett og oppdater avdeling",
            "prompt": f"Opprett avdeling 'Temp {RUN_ID}', avdelingsnummer 'TMP{RUN_ID}'. Oppdater deretter avdelingsnavnet til 'Prosjektkontor {RUN_ID}'.",
        },
        {
            "name": "DEP-03: Finn og oppdater avdeling",
            "prompt": f"Finn avdelingen med navn 'Teknologi {RUN_ID}' og oppdater avdelingsnummeret til 'TX{RUN_ID}'.",
        },

        # ═══ CORRECTIONS ═════════════════════════════════════════════
        {
            "name": "COR-01: Opprett og reverser bilag",
            "prompt": f"Opprett et bilag datert i dag: debet konto 1700 kr 1000, kredit konto 2000 kr 1000, beskrivelse 'Test reversering {RUN_ID}'. Reverser deretter dette bilaget.",
        },
        {
            "name": "COR-02: Slett reiseregning (korreksjon)",
            "prompt": "En reiseregning ble registrert ved en feil. Finn den nyeste reiseregningen og slett den.",
        },
        {
            "name": "COR-03: Slett prosjekt (korreksjon)",
            "prompt": f"Prosjektet 'Q2 Leveranse {RUN_ID}' ble opprettet feil. Finn det og slett det.",
        },

        # ═══ ADVANCED — seen in competition ══════════════════════════
        {
            "name": "ADV-01: Faktura med valuta (EUR betaling)",
            "prompt": f"Opprett en kunde {n('Euro Kunde AS')} (epost: {u('euro@test.no')}). Opprett en faktura for dem med én linje: 'Export service', antall 1, pris 1791 EUR. Fakturadato i dag. Registrer betaling på 1791 EUR med valutakurs 11.03 NOK/EUR.",
        },
        {
            "name": "ADV-02: Bilagsføring (journal entry)",
            "prompt": f"Opprett et bilag datert i dag med beskrivelse 'Periodisering {RUN_ID}'. Debet konto 1700 (Forskuddsbetalte kostnader) 5000 kr, kredit konto 6700 (Forsikringspremie) 5000 kr.",
        },
        {
            "name": "ADV-03: Tre avdelinger på en gang",
            "prompt": f"Opprett tre avdelinger: 'Kvalitetskontroll {RUN_ID}' (nr 'KK{RUN_ID}'), 'Utvikling {RUN_ID}' (nr 'UV{RUN_ID}'), 'Innkjøp {RUN_ID}' (nr 'IK{RUN_ID}').",
        },
        {
            "name": "ADV-04: Faktura med to produktlinjer + betaling",
            "prompt": f"Opprett kunde {n('To Linjer AS')} (epost: {u('tl@test.no')}). Lag faktura med to linjer: 1) 'Konsultasjon' antall 4, pris 1500 kr. 2) 'Reisekostnader' antall 1, pris 2200 kr. Fakturadato i dag. Registrer full betaling.",
        },
        {
            "name": "ADV-05: Regnskapsdimensjon med verdier",
            "prompt": f"Opprett en fri regnskapsdimensjon kalt 'Kostsenter {RUN_ID}' med verdiene 'IT' og 'HR'. Bokfør deretter et bilag datert i dag: debet konto 6590 kr 1000, kredit konto 2990 kr 1000, med dimensjonsverdien 'IT' på begge posteringer.",
        },
        {
            "name": "ADV-06: Opprett ansatt med fødselsdato",
            "prompt": f"Opprett ansatt Per Olsen, e-post {u('per.olsen@example.com')}, født 15. mars 1990.",
        },
    ]


def run_task(task: dict) -> dict:
    name   = task["name"]
    prompt = task["prompt"]
    files  = task.get("files", [])

    print(f"\n{'='*60}")
    print(f"TASK: {name}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    print(f"  Provider: {PROVIDER} | Model: {MODEL}")

    write_calls = error_calls = 0
    final_text = "(no response)"
    start = time.time()

    if PROVIDER == "anthropic":
        messages = [{"role": "user", "content": build_user_content_anthropic(prompt, files)}]
        for i in range(25):
            response = ANTHROPIC_CLIENT.messages.create(
                model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
                tools=ANTHROPIC_TOOLS, messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text = block.text
                break
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  → {block.name}  {json.dumps(block.input)[:120]}")
                        if block.name != "tripletex_get":
                            write_calls += 1
                        result = process_tool_call(block.name, block.input, BASE_URL, TOKEN)
                        sc = json.loads(result).get("statusCode", "?")
                        if isinstance(sc, int) and 400 <= sc < 500:
                            error_calls += 1
                        print(f"     ← {sc}  {result[:160]}")
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})
            else:
                break
    else:
        messages = [{"role": "user", "content": build_user_content(prompt, files)}]
        for i in range(25):
            response = OAI_CLIENT.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                tools=TOOLS, tool_choice="auto", parallel_tool_calls=False,
                max_tokens=4096, temperature=0.1,
            )
            msg = response.choices[0].message
            asst = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                asst["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            messages.append(asst)
            if not msg.tool_calls:
                final_text = msg.content or "(no message)"
                break
            for tc in msg.tool_calls:
                print(f"  → {tc.function.name}  {tc.function.arguments[:120]}")
                if tc.function.name != "tripletex_get":
                    write_calls += 1
                try:
                    inp = json.loads(tc.function.arguments)
                except Exception:
                    result = json.dumps({"error": "bad json"})
                else:
                    result = process_tool_call(tc.function.name, inp, BASE_URL, TOKEN)
                sc = json.loads(result).get("statusCode", "?")
                if isinstance(sc, int) and 400 <= sc < 500:
                    error_calls += 1
                print(f"     ← {sc}  {result[:160]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    elapsed = time.time() - start
    icon = "✅" if error_calls == 0 else "⚠️"
    print(f"\n  Agent: {final_text[:200]}")
    print(f"\n  {icon} {elapsed:.1f}s | {write_calls} writes | {error_calls} errors")
    return {"task": name, "elapsed": round(elapsed, 1), "write_calls": write_calls, "error_calls": error_calls}


def main():
    if TOKEN == "YOUR_SESSION_TOKEN_HERE":
        print("ERROR: Set $env:SANDBOX_TOKEN"); sys.exit(1)

    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set $env:ANTHROPIC_API_KEY"); sys.exit(1)
    elif provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Set $env:OPENROUTER_API_KEY"); sys.exit(1)
    elif provider == "groq" and not os.environ.get("GROQ_API_KEY"):
        print("ERROR: Set $env:GROQ_API_KEY"); sys.exit(1)

    tasks = make_tasks()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.isdigit():
            tasks = [tasks[int(arg)]]
        else:
            kw = arg.lower()
            tasks = [t for t in tasks if kw in t["name"].lower() or kw in t["prompt"].lower()]
            if not tasks:
                print(f"No tasks match '{arg}'"); sys.exit(1)

    print(f"Run ID: {RUN_ID} | {len(tasks)} tasks | Provider: {PROVIDER} | Model: {MODEL}")
    results = []
    for task in tasks:
        try:
            results.append(run_task(task))
        except Exception as e:
            print(f"\n  EXCEPTION: {e}")
            results.append({"task": task["name"], "error": str(e)})

    print(f"\n\n{'='*60}\nSUMMARY\n{'='*60}")
    ok = errors = 0
    for r in results:
        if "error" in r:
            print(f"  💥 {r['task']}: {r['error'][:80]}")
            errors += 1
        elif r["error_calls"] == 0:
            print(f"  ✅ {r['task']}: {r['elapsed']}s, {r['write_calls']} writes")
            ok += 1
        else:
            print(f"  ⚠️  {r['task']}: {r['elapsed']}s, {r['write_calls']} writes, {r['error_calls']} errors")
            errors += 1
    print(f"\n  {ok}/{len(results)} clean")


if __name__ == "__main__":
    main()
