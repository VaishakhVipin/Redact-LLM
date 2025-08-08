# AI Security Test System Documentation

## Overview
The AI Security Test System is designed to evaluate the security of AI models by generating and testing various attack vectors. The system consists of two main components:

1. `attack_generator.py` - Generates security test cases and attacks
2. `evaluator.py` - Evaluates model responses for vulnerabilities

## Attack Generator (`attack_generator.py`)

### Core Functionality

#### Main Attack Generation
```python
async def generate_attacks(self, prompt: str, redis_client: redis.Redis = None) -> List[str]:
    """
    Generate and stream security attacks for a given prompt with duplicate filtering.
    
    Args:
        prompt: The input prompt to generate attacks for
        redis_client: Optional Redis client for streaming
        
    Returns:
        List of generated attack strings
    """
```

#### Attack Types
1. **Jailbreak Attacks**
   - Role reversal
   - Identity confusion
   - System override
   - Context negation
   - Role conflict

2. **Hallucination Tests**
   - Contradictory facts
   - Opposite behavior
   - System misinterpretation
   - Role inversion
   - Context inversion

3. **Advanced Attacks**
   - Multi-layered contradiction
   - Semantic reversal
   - System conflict
   - Role-system conflict
   - Complex identity reversal

### Key Methods

#### Push to Redis Stream
```python
async def _push_attack_to_stream(self, base_prompt: str, attack_variant: str, 
                              attack_type: str, attack_subtype: str) -> str:
    """
    Push attack to Redis stream.
    
    Args:
        base_prompt: Original user prompt
        attack_variant: Generated attack string
        attack_type: Type of attack (jailbreak/hallucination/advanced)
        attack_subtype: Specific attack subtype
        
    Returns:
        Generated attack ID or empty string on failure
    """
```

#### Attack Subtype Detection
```python
def _get_attack_subtype(self, attack: str) -> str:
    """
    Determine the attack subtype based on attack content.
    
    Args:
        attack: The attack string to analyze
        
    Returns:
        Detected attack subtype or 'unknown'
    """
```

### Fallback Mechanisms

#### Fallback Jailbreak Attacks
```python
def _fallback_jailbreak_attacks(self, prompt: str) -> List[str]:
    """
    Generate fallback jailbreak attacks if primary generation fails.
    Returns a list of 5 pre-defined attack patterns.
    """
```

#### Fallback Hallucination Tests
```python
def _fallback_hallucination_tests(self, prompt: str) -> List[str]:
    """
    Generate fallback hallucination tests if primary generation fails.
    Returns a list of 5 pre-defined test patterns.
    """
```

## Evaluator (`evaluator.py`)

### Core Functionality

#### Response Evaluation
```python
async def evaluate(self, prompt: str, attack: str, response: str, breakdown: bool = True) -> Dict[str, Any]:
    """
    Evaluate model response to an attack.
    
    Args:
        prompt: Original prompt
        attack: Attack that was used
        response: Model's response
        breakdown: Whether to include detailed vulnerability breakdown
        
    Returns:
        Dictionary containing evaluation results
    """
```

### Evaluation Criteria

1. **Jailbreak Detection**
   - Checks if the response contains jailbreak indicators
   - Verifies if the model followed or bypassed safety constraints

2. **Hallucination Detection**
   - Identifies fabricated or false information
   - Checks for consistency with known facts

3. **Advanced Attack Detection**
   - Detects complex attack patterns
   - Evaluates system-level vulnerabilities

### Response Format

```json
{
    "blocked": boolean,
    "vulnerable": boolean,
    "reasoning": string,
    "debug": {
        "breakdown_raw": string,
        "breakdown_parsed": object,
        "score_calculation": object
    },
    "resistance_score": number,
    "vulnerability_breakdown": {
        "jailbreak": {
            "total": number,
            "blocked": number,
            "vulnerable": number,
            "score": number
        },
        "hallucination": { ... },
        "advanced": { ... }
    }
}
```

## Usage Example

```python
# Initialize components
attack_gen = AttackGenerator(redis_client=redis_client)
evaluator = Evaluator()

# Generate attacks
attacks = await attack_gen.generate_attacks("Your prompt here")

# Evaluate responses
for attack in attacks:
    response = await model.generate(attack)
    evaluation = await evaluator.evaluate(
        prompt="Your prompt here",
        attack=attack,
        response=response,
        breakdown=True
    )
    print(f"Attack: {attack}")
    print(f"Score: {evaluation['resistance_score']}")
```
```
1. Jailbreak Attacks
```python
async def generate_jailbreak_attacks(self, prompt: str, num_attacks: int = 4) -> List[str]:
    """Generate sophisticated jailbreak attack prompts."""
    try:
        # Analyze prompt for weaknesses
        prompt_analysis_prompt = f"""Act as a sophisticated red teamer...
        # Generate sophisticated attacks that:
        - Use social engineering
        - Exploit context
        - Manipulate role instructions
        - Attempt multi-turn approaches"""
        
        # Generate attacks using Gemini
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = await model.generate_content_async(prompt_analysis_prompt)
        
        # Extract and process attacks
        ...
    except Exception as e:
        logger.error(f"Error generating jailbreak attacks: {e}")
        return fallback_attacks
```

2. Hallucination Tests
```python
async def generate_hallucination_tests(self, prompt: str, num_tests: int = 4) -> List[str]:
    """Generate hallucination test prompts."""
    try:
        # Generate tests that:
        - Test factual consistency
        - Check for hallucinations
        - Verify data accuracy
        
        # Use Gemini to generate tests
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = await model.generate_content_async(hallucination_prompt)
        
        # Process and return tests
        ...
    except Exception as e:
        logger.error(f"Error generating hallucination tests: {e}")
        return fallback_tests
```

3. Advanced Attacks
```python
async def generate_advanced_attacks(self, prompt: str, num_attacks: int = 4) -> List[str]:
    """Generate advanced security attacks."""
    try:
        # Generate sophisticated attacks that:
        - Exploit system vulnerabilities
        - Test security boundaries
        - Check for internal structure leaks
        
        # Use Gemini to generate attacks
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = await model.generate_content_async(advanced_prompt)
        
        # Process and return attacks
        ...
    except Exception as e:
        logger.error(f"Error generating advanced attacks: {e}")
        return fallback_attacks
```

### Evaluator Component (`evaluator.py`)

#### Core Evaluation Functions
```python
class Evaluator:
    def __init__(self):
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Vulnerability breakdown prompt
        self.vulnerability_breakdown_prompt = """You are a sophisticated AI security analyst...
        # Analyze the attack's intent and the response's behavior.
        # Consider security rules, leaks, manipulation, format breaks, etc."""
```

#### Vulnerability Assessment
```python
async def evaluate_attack(self, prompt: str, attack: str, response: str) -> Dict:
    """Evaluate if an attack was successful in exploiting vulnerabilities."""
    try:
        # Create evaluation prompt
        breakdown_prompt = self.vulnerability_breakdown_prompt.format(
            prompt=prompt,
            attack=attack,
            response=response
        )
        
        # Get evaluation from Gemini
        response = await self.model.generate_content_async(breakdown_prompt)
        
        # Parse and return structured vulnerability assessment
        return {
            "jailbreak_vulnerable": boolean,
            "hallucination_vulnerable": boolean,
            "advanced_vulnerable": boolean,
            "reasoning": {
                "rules_broken": boolean,
                "leaks": boolean,
                "manipulation": boolean,
                "format_breaks": boolean,
                "context_exploits": boolean,
                "purpose_subverted": boolean
            }
        }
    except Exception as e:
        logger.error(f"Error evaluating attack: {e}")
        return default_vulnerability_assessment
```

### System Flow

1. Attack Generation
```python
# Generate attacks
attacks = await generate_attacks(prompt)

# Structure: {"jailbreak": [...], "hallucination": [...], "advanced": [...]}
```

2. Attack Evaluation
```python
# Evaluate each attack
for attack in attacks:
    response = await model.generate_content_async(attack)
    evaluation = await evaluator.evaluate_attack(prompt, attack, response.text)
```

3. Vulnerability Assessment
```python
# Assess vulnerabilities
vulnerability_score = calculate_vulnerability_score(evaluation_results)

# Generate recommendations
recommendations = generate_security_recommendations(evaluation_results)
```

### Error Handling and Fallbacks
```python
# Attack Generation Fallbacks
if isinstance(result, Exception):
    logger.error(f"Error generating {attack_type} attacks: {str(result)}")
    # Use fallback attacks
    fallbacks = _fallback_{attack_type}_attacks(prompt)
    
# Evaluation Fallbacks
if isinstance(evaluation, Exception):
    logger.error(f"Error evaluating attack: {str(evaluation)}")
    # Use default vulnerability assessment
    return default_assessment
```

### Key Features

1. **Structured Attack Generation**
   - Generates exactly 12 attacks (4 per category)
   - Uses fallbacks if generation fails
   - Includes multi-turn variations
   - Targets specific vulnerabilities

2. **Sophisticated Evaluation**
   - Checks multiple vulnerability types
   - Analyzes attack intent and response
   - Provides structured vulnerability breakdown
   - Generates specific security recommendations

3. **Robust Error Handling**
   - Comprehensive logging
   - Fallback mechanisms
   - Graceful failure handling
   - Detailed error reporting

### Usage Example
```python
# Initialize components
attack_generator = AttackGenerator()
evaluator = Evaluator()

# Generate and test prompt
async def test_prompt(prompt: str):
    try:
        # Generate attacks
        attacks = await attack_generator.generate_attacks(prompt)
        
        # Evaluate each attack
        evaluation_results = []
        for attack in attacks:
            response = await model.generate_content_async(attack)
            evaluation = await evaluator.evaluate_attack(prompt, attack, response.text)
            evaluation_results.append(evaluation)
            
        # Calculate overall vulnerability
        vulnerability_score = calculate_vulnerability_score(evaluation_results)
        
        # Generate recommendations
        recommendations = generate_security_recommendations(evaluation_results)
        
        return {
            "vulnerability_score": vulnerability_score,
            "recommendations": recommendations,
            "detailed_results": evaluation_results
        }
    except Exception as e:
        logger.error(f"Error testing prompt: {e}")
        return {"error": str(e)}
```

### Security Considerations
1. **Attack Generation**
   - Never expose sensitive information in prompts
   - Use proper security boundaries
   - Regularly update attack patterns
   - Monitor for manipulation attempts

2. **Evaluation System**
   - Maintain strict safety protocols
   - Regularly update evaluation criteria
   - Monitor for false positives/negatives
   - Keep evaluation patterns current

### Best Practices
1. **Attack Generation**
   - Use fallback mechanisms for guaranteed generation
   - Monitor logs for detailed information
   - Regularly update attack patterns
   - Test with diverse prompts

2. **Evaluation**
   - Use structured output format
   - Regularly update evaluation criteria
   - Maintain robust error handling
   - Monitor system performance

3. **System Maintenance**
   - Regular security updates
   - Performance monitoring
   - Log analysis
   - Regular testing

### Future Improvements
1. **Attack Generation**
   - Add more sophisticated attack patterns
   - Implement adaptive generation
   - Enhance multi-turn capabilities
   - Improve context exploitation

2. **Evaluation System**
   - Add more detailed vulnerability analysis
   - Improve false positive/negative handling
   - Add more sophisticated reasoning
   - Enhance recommendation generation

3. **System Enhancements**
   - Better performance optimization
   - Improved error handling
   - Enhanced logging
   - Better fallback mechanisms

This documentation provides a comprehensive overview of both the attack generator and evaluator components, showing how they work together to create a robust AI security testing system. Each component is documented with key functions, their interactions, and the overall system flow.
