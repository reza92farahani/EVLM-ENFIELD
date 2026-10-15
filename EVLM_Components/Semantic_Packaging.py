import json
import re
import os
import argparse
import time


class InspectionAnalyzer:
    """
    Strict VLM Parser using Sentence-Level Logic.

    - Defects use strict negation filtering
    - Components use keyword presence only
    - Reads multiple JSON files
    - Saves structured results
    """

    def __init__(self):
        self.schema = {
            "image_id": "",
            "meta": {
                "prompt_type": "unknown",
                "prompt_text": "",
                "requires_visual_evidence": False
            },
            "findings": {
                "summary": "",
                "defects": [],
                "components": []
            },
            "raw_vlm_output": ""
        }

        # ---- Defect Categories (STRICT) ----
        self.defect_map = {
            "corrosion": ["rust", "corro", "oxidiz"],
            "vegetation": ["vegetation", "tree", "branch", "growth", "encroach"],
            "mechanical": ["break", "broken", "crack", "damage", "missing", "snap", "loose"],
            "sagging": ["sag", "droop", "low hang"]
        }

        # ---- Component Categories (NON-STRICT) ----
        self.component_map = {
            "tower": ["tower", "pole", "structure"],
            "insulator": ["insulator"],
            "conductor": ["wire", "conductor", "cable"],
            "crossarm": ["crossarm"],
            "transformer": ["transformer"]
        }

    # ---------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------

    def _split_into_sentences(self, text):
        return re.split(r'[.!\n]+', text)

    def _is_sentence_positive(self, sentence, keywords):
        """
        STRICT logic:
        Keyword present AND no negation words in same sentence.
        """
        s_lower = sentence.lower().strip()
        if not s_lower:
            return False

        has_keyword = any(k in s_lower for k in keywords)
        if not has_keyword:
            return False

        negations = [
            "no ", "not ", "n't ", "none ", "neither ", "nor ",
            "free of", "free from", "without", "lack of", "absent",
            "clean", "clear of", "0", "zero"
        ]

        is_negated = any(neg in s_lower for neg in negations)

        return not is_negated

    def _contains_keyword(self, sentence, keywords):
        """
        NON-STRICT logic (for components).
        Only checks keyword presence.
        """
        s_lower = sentence.lower().strip()
        return any(k in s_lower for k in keywords)

    def determine_intent(self, prompt):
        p = prompt.lower()

        if any(x in p for x in ["defect", "damage", "rust", "corrosion",
                                "vegetation", "broken", "crack", "issue"]):
            return "defect_search"

        if "where" in p or "location" in p:
            return "localization"

        if any(x in p for x in ["rain", "fog", "glare", "blur", "weather"]):
            return "visibility_check"

        if "list" in p or "components" in p or "what" in p:
            return "enumeration"

        return "general_description"

    # ---------------------------------------------------
    # Core Analysis
    # ---------------------------------------------------

    def analyze_image(self, entry):
        report = json.loads(json.dumps(self.schema))

        report["image_id"] = entry.get('frame_name', '')
        report["meta"]["prompt_text"] = entry.get('prompt', '')
        report["raw_vlm_output"] = entry.get('result', '')

        intent = self.determine_intent(report["meta"]["prompt_text"])
        report["meta"]["prompt_type"] = intent

        sentences = self._split_into_sentences(report["raw_vlm_output"])

        # ---------------------------------------------------
        # COMPONENT DETECTION (Always Run)
        # ---------------------------------------------------
        found_components = []

        for sentence in sentences:
            for ctype, keywords in self.component_map.items():
                if self._contains_keyword(sentence, keywords):
                    found_components.append(ctype)

        report["findings"]["components"] = list(set(found_components))

        # ---------------------------------------------------
        # DEFECT SEARCH (STRICT)
        # ---------------------------------------------------
        if intent == "defect_search":
            found_defects = []

            for sentence in sentences:
                for dtype, keywords in self.defect_map.items():
                    if self._is_sentence_positive(sentence, keywords):
                        found_defects.append(dtype)

            found_defects = list(set(found_defects))
            report["findings"]["defects"] = found_defects

            if found_defects:
                report["findings"]["summary"] = f"DEFECT CONFIRMED: {', '.join(found_defects)}"
                report["meta"]["requires_visual_evidence"] = True
            else:
                report["findings"]["summary"] = "No defects detected."
                report["meta"]["requires_visual_evidence"] = False

        # ---------------------------------------------------
        # LOCALIZATION / ENUMERATION
        # ---------------------------------------------------
        elif intent in ["localization", "enumeration"]:
            relevant_sentence_found = False

            for sentence in sentences:
                s_lower = sentence.lower()

                if "not visible" in s_lower or "cannot see" in s_lower or "no " in s_lower:
                    continue

                if any(x in s_lower for x in
                       ["top", "left", "right", "bottom", "center",
                        "tower", "insulator", "conductor"]):
                    relevant_sentence_found = True
                    report["findings"]["summary"] = sentence.strip()

            report["meta"]["requires_visual_evidence"] = relevant_sentence_found

        # ---------------------------------------------------
        # VISIBILITY CHECK
        # ---------------------------------------------------
        elif intent == "visibility_check":
            issues = []

            for sentence in sentences:
                if self._is_sentence_positive(sentence,
                                              ["fog", "rain", "glare", "blur", "haze"]):
                    issues.append(sentence.strip())

            if issues:
                report["findings"]["summary"] = "Visibility issues detected."
                report["meta"]["requires_visual_evidence"] = True
            else:
                report["findings"]["summary"] = "Clear conditions."
                report["meta"]["requires_visual_evidence"] = False

        # ---------------------------------------------------
        # GENERAL DESCRIPTION
        # ---------------------------------------------------
        else:
            lower_output = report["raw_vlm_output"].lower()

            is_bad_image = any(x in lower_output for x in
                               ["nothing visible", "image is black",
                                "too blurry", "cannot determine"])

            report["findings"]["summary"] = report["raw_vlm_output"][:120]
            report["meta"]["requires_visual_evidence"] = False #not is_bad_image

        return report


# ==========================================================
# RUNNER
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="vlm_results")
    parser.add_argument("--output_dir", default="semantic_report")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        exit()

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    analyzer = InspectionAnalyzer()

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    if not json_files:
        print("No JSON files found.")
        exit()

    print(f"Found {len(json_files)} JSON files. Processing...\n")

    total_send_count = 0
    total_images = 0

    for filename in json_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as f:
            raw_data = json.load(f)

        final_reports = []
        send_count = 0

        print(f"Processing file: {filename} ({len(raw_data)} images)")

        for item in raw_data:
            structured = analyzer.analyze_image(item)
            final_reports.append(structured)

            total_images += 1
            if structured["meta"]["requires_visual_evidence"]:
                send_count += 1
                total_send_count += 1

        with open(output_path, 'w') as f:
            json.dump(final_reports, f, indent=2)

        print(f"  → {send_count} images marked for transmission.\n")
    print("duration", time.time()-start_time)
    print("ALL FILES COMPLETE.")
    print(f"Total images processed: {total_images}")
    print(f"Total images marked for transmission: {total_send_count}")
