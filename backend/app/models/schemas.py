from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field

class Recommendation(BaseModel):
    """Security recommendation with severity level"""
    category: Literal["SECURITY", "CLARITY", "PERFORMANCE", "OTHER"]
    action: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]

class AttackEvaluation(BaseModel):
    """Evaluation results for a single attack type"""
    blocked: bool
    reasoning: str
    confidence: float = Field(..., ge=0, le=1)
    category: str

class VulnerabilityBreakdown(BaseModel):
    """Vulnerability statistics for a specific category"""
    total: int = Field(..., ge=0)
    blocked: int = Field(..., ge=0)
    vulnerable: int = Field(..., ge=0)
    score: float = Field(..., ge=0, le=100)

class TestEvaluationResponse(BaseModel):
    """Response model for test evaluation endpoint"""
    breakdown: AttackEvaluation
    debug: Dict[str, Any] = {}
    prompt: str
    raw_result: str
    test_id: str
    original_prompt: str
    resistance_score: float = Field(..., ge=0, le=100)
    total_attacks: int = Field(..., ge=0)
    vulnerability_breakdown: Dict[str, VulnerabilityBreakdown]
    recommendations: List[Recommendation]
    attacks: List[str]

class AttackGenerationResponse(BaseModel):
    """Response model for attack generation"""
    attacks: List[str]
    count: int = Field(..., ge=0)
    categories: Dict[str, int]
    prompt: Optional[str] = None

class ResistanceTestResponse(BaseModel):
    """Response model for resistance testing"""
    test_id: str
    original_prompt: str
    resistance_score: float = Field(..., ge=0, le=100)
    total_attacks: int = Field(..., ge=0)
    vulnerability_breakdown: Dict[str, VulnerabilityBreakdown]
    recommendations: List[Recommendation]
    attacks: List[str]
    target_response: Optional[str] = None