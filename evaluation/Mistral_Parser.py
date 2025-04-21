import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union

class LLMJsonParser:
    """
    A parser for extracting JSON from LLM outputs that may contain additional text,
    malformed JSON, or incomplete JSON, including arrays that are cut off.
    """
    
    @staticmethod
    def extract_json_candidates(text: str) -> List[str]:
        """
        Extract potential JSON strings from text by looking for patterns of balanced braces.
        
        Args:
            text: The text containing potential JSON.
            
        Returns:
            List of candidate JSON strings.
        """
        candidates = []
        
        # Look for text starting with { and extract as much valid JSON structure as possible
        json_start_pattern = re.compile(r'{')
        for match in json_start_pattern.finditer(text):
            start_idx = match.start()
            # Get the substring from the start to the end of the text
            potential_json = text[start_idx:]
            candidates.append(potential_json)
        
        return candidates
    
    @staticmethod
    def fix_common_json_errors(json_str: str) -> str:
        """
        Attempt to fix common JSON formatting errors.
        
        Args:
            json_str: The potentially malformed JSON string.
            
        Returns:
            Corrected JSON string.
        """
        # Replace single quotes with double quotes (but not within quoted strings)
        fixed = ''
        in_double_quotes = False
        for i, char in enumerate(json_str):
            if char == '"' and (i == 0 or json_str[i-1] != '\\'):
                in_double_quotes = not in_double_quotes
                fixed += char
            elif char == "'" and not in_double_quotes:
                fixed += '"'
            else:
                fixed += char
        
        # Fix missing commas between objects in arrays
        fixed = re.sub(r'}\s*{', '},{', fixed)
        
        # Fix trailing commas
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Ensure property names are double-quoted
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed)
        
        return fixed
    
    @staticmethod
    def complete_incomplete_json(json_str: str) -> str:
        """
        Complete incomplete JSON by analyzing the structure and adding missing 
        closing elements. Handles nested structures and arrays.
        
        Args:
            json_str: The incomplete JSON string.
            
        Returns:
            Completed JSON string.
        """
        # Track opening and closing symbols
        open_stack = []
        expecting_comma_or_close = False
        in_string = False
        escape_next = False
        
        # Track if we're in an object where we need to make sure keys have values
        key_value_pairs = []
        current_key = None
        has_colon = False
        
        for i, char in enumerate(json_str):
            # Handle string boundaries and escaping
            if char == '"' and not escape_next:
                in_string = not in_string
                if not in_string and len(open_stack) > 0 and open_stack[-1] == '{' and current_key is None:
                    # End of a key in an object
                    current_key = True
            elif char == '\\' and in_string:
                escape_next = True
                continue
            
            # Skip processing structural elements if in a string
            if in_string:
                escape_next = False
                continue
            
            # Process structural JSON elements
            if char == '{':
                open_stack.append('{')
                expecting_comma_or_close = False
                key_value_pairs.append((current_key, has_colon))
                current_key = None
                has_colon = False
            elif char == '[':
                open_stack.append('[')
                expecting_comma_or_close = False
            elif char == ':':
                if current_key:
                    has_colon = True
            elif char == ',':
                expecting_comma_or_close = False
                current_key = None
                has_colon = False
            elif char == '}':
                if open_stack and open_stack[-1] == '{':
                    open_stack.pop()
                    current_key, has_colon = key_value_pairs.pop() if key_value_pairs else (None, False)
                    expecting_comma_or_close = True
            elif char == ']':
                if open_stack and open_stack[-1] == '[':
                    open_stack.pop()
                    expecting_comma_or_close = True
        
        # Complete the JSON structure
        completed = json_str
        
        # If we have a partial key without a value, add a null value
        if current_key and has_colon and completed[-1] not in ']}':
            completed += 'null'
        
        # If we have a key without a colon, add it with a null value
        if current_key and not has_colon:
            completed += ': null'
        
        # If we ended in the middle of an array item, we need to close the item properly
        if open_stack and open_stack[-1] == '[' and expecting_comma_or_close:
            pass  # The item is complete, just need to close the array
        elif open_stack and open_stack[-1] == '[' and not expecting_comma_or_close:
            # We might be in the middle of an array element
            if completed[-1] not in '{[,':
                pass  # Just close the array
        
        # Close any remaining open structures
        for opener in reversed(open_stack):
            if opener == '{':
                completed += '}'
            elif opener == '[':
                completed += ']'
        
        return completed
    
    @staticmethod
    def parse_json(text: str) -> Tuple[Optional[Any], List[Any]]:
        """
        Parse JSON from text that may contain additional content or malformed JSON.
        
        Args:
            text: Text that may contain JSON.
            
        Returns:
            Tuple containing:
            - The best parsed JSON object (or None if no valid JSON found)
            - List of all valid JSON objects found
        """
        # First try to parse the entire text as JSON
        try:
            parsed = json.loads(text)
            return parsed, [parsed]
        except json.JSONDecodeError:
            pass
        
        # Extract candidate JSON strings
        candidates = LLMJsonParser.extract_json_candidates(text)
        
        # Try to parse each candidate
        valid_jsons = []
        
        for candidate in candidates:
            # Try parsing as is
            try:
                parsed = json.loads(candidate)
                valid_jsons.append(parsed)
                continue
            except json.JSONDecodeError:
                pass
            
            # Try fixing common errors
            fixed = LLMJsonParser.fix_common_json_errors(candidate)
            try:
                parsed = json.loads(fixed)
                valid_jsons.append(parsed)
                continue
            except json.JSONDecodeError:
                pass
            
            # Try completing incomplete JSON
            completed = LLMJsonParser.complete_incomplete_json(fixed)
            try:
                parsed = json.loads(completed)
                valid_jsons.append(parsed)
            except json.JSONDecodeError:
                # If still failing, try a different approach: incrementally parse
                try:
                    result = LLMJsonParser.incremental_parse(candidate)
                    if result:
                        valid_jsons.append(result)
                except Exception:
                    pass
        
        # Return the most complex valid JSON as the primary result
        if valid_jsons:
            # Use the JSON with the most keys as the "best" one
            best_json = max(valid_jsons, key=lambda x: str(x).count(':'))
            return best_json, valid_jsons
        
        return None, []
    
    @staticmethod
    def incremental_parse(text: str) -> Optional[Any]:
        """
        Incrementally parse the JSON by trying to find the largest valid subset.
        
        Args:
            text: Potential JSON text.
            
        Returns:
            Parsed JSON object or None.
        """
        # For deeply nested structures, we need to be smarter about completion
        if text.startswith('{'):
            # Find all array and object closing positions
            stack = []
            closing_positions = []
            
            for i, char in enumerate(text):
                if char in '{[':
                    stack.append(char)
                elif char == '}' and stack and stack[-1] == '{':
                    stack.pop()
                    if not stack:  # We've reached a complete object
                        closing_positions.append(i)
                elif char == ']' and stack and stack[-1] == '[':
                    stack.pop()
                    if not stack:  # We've reached a complete array
                        closing_positions.append(i)
            
            # Try to parse increasingly larger chunks
            for pos in sorted(closing_positions):
                try:
                    return json.loads(text[:pos+1])
                except json.JSONDecodeError:
                    pass
            
            # If no complete objects found, manually complete the structure
            return LLMJsonParser.manual_completion(text)
        
        return None
    
    @staticmethod
    def manual_completion(text: str) -> Optional[Any]:
        """
        Manually attempt to complete a partial JSON structure by analyzing the pattern.
        
        Args:
            text: Partial JSON text.
            
        Returns:
            Parsed JSON object or None.
        """
        # For this specific case, we know we're dealing with an array of objects inside "topical_attitudes"
        if '"topical_attitudes":' in text and '[{' in text:
            # Identify how many objects we have started but not completed
            parts = text.split('{"source":')
            
            if len(parts) <= 1:
                return None
            
            # Rebuild the structure with proper completion
            reconstructed = parts[0] + '{"source":'
            
            for i, part in enumerate(parts[1:]):
                if i < len(parts) - 2:  # Not the last part
                    reconstructed += part
                else:  # Last part needs completion
                    # Count open braces to determine nesting level
                    open_braces = part.count('{')
                    close_braces = part.count('}')
                    missing_braces = open_braces - close_braces
                    
                    reconstructed += part
                    if missing_braces > 0:
                        reconstructed += '}' * missing_braces
                    
                    # Check if we need to close the array and the topical_attitudes object
                    if '"topical_attitudes": [' in reconstructed and reconstructed.count('[') > reconstructed.count(']'):
                        reconstructed += ']}'
            
            try:
                return json.loads(reconstructed)
            except json.JSONDecodeError:
                pass
        
        return None


# Example usage
def parse_llm_output(text: str) -> Dict:
    """
    Parse JSON from LLM output and extract structured data.
    
    Args:
        text: LLM output text.
        
    Returns:
        Extracted JSON data as a dictionary.
    """
    parser = LLMJsonParser()
    best_json, all_jsons = parser.parse_json(text)
    
    return {
        "best_result": best_json,
        "all_results": all_jsons,
        "result_count": len(all_jsons)
    }