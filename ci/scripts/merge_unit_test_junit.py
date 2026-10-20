"""Merge serial and parallel pytest reports without discarding either phase."""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


def merge_reports(paths: list[Path], output: Path) -> bool:
    root = ET.Element("testsuites")
    complete = True
    for path in paths:
        try:
            source = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as error:
            complete = False
            suite = ET.SubElement(root, "testsuite", name=path.stem, tests="1", errors="1")
            missing = isinstance(error, FileNotFoundError)
            name = "missing_phase_report" if missing else "invalid_phase_report"
            case = ET.SubElement(suite, "testcase", name=name, classname="ci")
            message = (
                f"pytest did not write {path.name}"
                if missing
                else f"Cannot read {path.name}: {error}"
            )
            ET.SubElement(case, "error", message=message)
            continue
        if source.tag == "testsuite":
            root.append(source)
        else:
            root.extend(source.findall("testsuite"))
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    if not merge_reports(args.reports, args.output):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
