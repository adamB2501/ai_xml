import json
import os
import sys

from extraction.ner_extractor import extract_fields
from extraction.pdf_reader import extract_text_auto
from mapping.xml_builder import write_teif_xml
from validation.validate_xml import print_report, validate_teif_xml


def process_invoice(pdf_path, output_dir="data/output"):
    """Runs the full pipeline: PDF -> text -> NER fields -> JSON + TEIF XML
    -> structural/arithmetic validation of that XML. Every stage that can
    produce partial or questionable output (extract_fields' other_fields,
    xml_builder's warnings, validate_xml's findings) surfaces that instead
    of hiding it -- see result["build_warnings"] / result["validation_findings"].
    """
    text = extract_text_auto(pdf_path)
    fields = extract_fields(text)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(output_dir, f"{filename}.json")
    xml_path = os.path.join(output_dir, f"{filename}.xml")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, ensure_ascii=False)

    build_warnings = write_teif_xml(fields, xml_path)
    validation_findings = validate_teif_xml(xml_path)

    return {
        "fields": fields,
        "json_path": json_path,
        "xml_path": xml_path,
        "build_warnings": build_warnings,
        "validation_findings": validation_findings,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_invoice.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = process_invoice(pdf_path)

    print(json.dumps(result["fields"], indent=2, ensure_ascii=False))
    print(f"\nSaved JSON to {result['json_path']}")
    print(f"Saved XML to {result['xml_path']}")

    if result["build_warnings"]:
        print(f"\n[xml_builder] {len(result['build_warnings'])} warning(s):")
        for w in result["build_warnings"]:
            print(f"  - {w}")

    print()
    print_report(result["validation_findings"])

    if any(f.severity == "error" for f in result["validation_findings"]):
        sys.exit(1)
