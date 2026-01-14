import json
from pathlib import Path

def validate_json_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return True, data, None
    except json.JSONDecodeError as e:
        return False, None, f"JSON syntax error: {e}"
    except FileNotFoundError:
        return False, None, f"File not found: {filepath}"

def validate_knowledge_base(data):
    errors = []
    warnings = []
    
    required_sections = ["product", "pricing", "policies"]
    for section in required_sections:
        if section not in data:
            errors.append(f"Missing required section: {section}")
    
    if "pricing" in data:
        for plan_name, plan_data in data["pricing"].items():
            if "price" not in plan_data:
                warnings.append(f"Plan '{plan_name}' missing 'price' field")
            if "features" not in plan_data and plan_name != "enterprise":
                warnings.append(f"Plan '{plan_name}' missing 'features' list")
    
    return errors, warnings

def validate_conversation_examples(data):
    errors = []
    warnings = []
    
    required_sections = ["intent_classification_examples", "rag_response_examples", 
                        "lead_qualification_examples", "complete_conversations"]
    
    for section in required_sections:
        if section not in data:
            warnings.append(f"Recommended section missing: {section}")
    
    if "intent_classification_examples" in data:
        valid_intents = ["greeting", "product_or_pricing", "high_intent_lead"]
        for i, example in enumerate(data["intent_classification_examples"]):
            if "intent" not in example:
                errors.append(f"Intent example {i} missing 'intent' field")
            elif example["intent"] not in valid_intents:
                errors.append(f"Intent example {i} has invalid intent: {example['intent']}")
            if "user" not in example:
                errors.append(f"Intent example {i} missing 'user' field")
    
    return errors, warnings

def validate_objection_library(data):
    errors = []
    warnings = []
    
    if "objections" not in data:
        errors.append("Missing 'objections' array")
        return errors, warnings
    
    for i, objection in enumerate(data["objections"]):
        required_fields = ["type", "keywords", "response"]
        for field in required_fields:
            if field not in objection:
                errors.append(f"Objection {i} ({objection.get('type', 'unknown')}) missing '{field}' field")
        
        if "keywords" in objection and not isinstance(objection["keywords"], list):
            errors.append(f"Objection {i} 'keywords' must be a list")
    
    return errors, warnings

def main():
    print("=" * 70)
    print("DATA VALIDATION REPORT")
    print("=" * 70)
    print()
    
    data_dir = Path("data")
    files_to_validate = {
        "knowledge_base.json": validate_knowledge_base,
        "conversation_examples.json": validate_conversation_examples,
        "objection_library.json": validate_objection_library
    }
    
    all_valid = True
    
    for filename, validator in files_to_validate.items():
        filepath = data_dir / filename
        print(f"Validating {filename}...")
        print("-" * 70)
        
        is_valid, data, error = validate_json_file(filepath)
        
        if not is_valid:
            print(f"❌ FAILED: {error}")
            all_valid = False
            print()
            continue
        
        errors, warnings = validator(data)
        
        if errors:
            print(f"❌ ERRORS FOUND ({len(errors)}):")
            for error in errors:
                print(f"   - {error}")
            all_valid = False
        else:
            print("✅ No errors found")
        
        if warnings:
            print(f"⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"   - {warning}")
        
        print()
    
    print("=" * 70)
    if all_valid:
        print("✅ ALL DATA FILES VALID")
    else:
        print("❌ VALIDATION FAILED - Please fix errors above")
    print("=" * 70)

if __name__ == "__main__":
    main()