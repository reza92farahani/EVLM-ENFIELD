import json
import os
import shutil
import argparse
from datetime import datetime
from PIL import Image


class MissionController:
    def __init__(self, raw_images_dir, output_dir):
        self.raw_dir = raw_images_dir
        self.pkg_dir = output_dir
        self.img_dir = os.path.join(output_dir, "images")

        # Reset output directory structure for this package
        if os.path.exists(self.pkg_dir):
            shutil.rmtree(self.pkg_dir)
        os.makedirs(self.img_dir)

        # Build image index (recursive scan once)
        self.image_index = {}
        print("Indexing images...")
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                # If duplicate filenames exist, first one found is used
                if file not in self.image_index:
                    self.image_index[file] = os.path.join(root, file)

        print(f"Indexed {len(self.image_index)} images.")

    def optimize_image(self, filename):
        src_path = self.image_index.get(filename)

        if not src_path or not os.path.exists(src_path):
            return None

        try:
            with Image.open(src_path) as img:
                # Resize if wider than 1920px
                if img.width > 1920:
                    ratio = 1920 / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((1920, new_height), Image.Resampling.LANCZOS)

                new_filename = os.path.splitext(filename)[0] + ".webp"
                dest_path = os.path.join(self.img_dir, new_filename)

                img.save(dest_path, "WEBP", quality=80, method=6)
                return new_filename

        except Exception as e:
            print(f"Error optimizing {filename}: {e}")
            return None

    def execute(self, report_path):
        if not os.path.exists(report_path):
            print(f"Error: Report {report_path} not found.")
            return

        with open(report_path, 'r') as f:
            full_data = json.load(f)

        print(f"Processing {len(full_data)} records from {os.path.basename(report_path)}...")

        final_mission_report = []
        images_sent_count = 0
        defects_count = 0

        for entry in full_data:
            record = entry.copy()

            if record["meta"]["requires_visual_evidence"]:
                optimized_name = self.optimize_image(entry['image_id'])

                if optimized_name:
                    record["transmission_status"] = {
                        "image_included": True,
                        "file_name": optimized_name,
                        "reason": "Visual Evidence Required"
                    }
                    images_sent_count += 1
                else:
                    record["transmission_status"] = {
                        "image_included": False,
                        "error": "Source file missing"
                    }

                if "DEFECT" in str(entry["findings"].get("summary", "")):
                    defects_count += 1
            else:
                record["transmission_status"] = {
                    "image_included": False,
                    "reason": "Not required (Negative/Clean Result)"
                }

            final_mission_report.append(record)

        final_package_json = {
            "mission_id": f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "stats": {
                "total_records": len(full_data),
                "images_transmitted": images_sent_count,
                "defects_flagged": defects_count
            },
            "records": final_mission_report
        }

        # Save JSON report for this package
        json_output_path = os.path.join(self.pkg_dir, "mission_full_report.json")
        with open(json_output_path, 'w') as f:
            json.dump(final_package_json, f, indent=2)

        # Create ZIP package
        #shutil.make_archive(self.pkg_dir, 'zip', self.pkg_dir)

        print(f"Package created: {self.pkg_dir}.zip | "
              f"Records: {len(full_data)}, "
              f"Images included: {images_sent_count}")


# --- RUNNER ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_dir", default="./semantic_report",
                        help="Folder with processed JSON reports")
    parser.add_argument("--images", default=".",
                        help="Folder with original high-res images (searched recursively)")
    parser.add_argument("--out", default="./transmission_packages",
                        help="Root output folder for packages")
    args = parser.parse_args()

    # Create root output folder if missing
    os.makedirs(args.out, exist_ok=True)

    # Iterate over all JSONs in the reports folder
    json_files = [f for f in os.listdir(args.reports_dir) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {args.reports_dir}.")
        exit()

    for json_file in json_files:
        report_path = os.path.join(args.reports_dir, json_file)
        package_name = os.path.splitext(json_file)[0]
        package_dir = os.path.join(args.out, package_name)

        controller = MissionController(args.images, package_dir)
        controller.execute(report_path)
