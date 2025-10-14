from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class EpistemicType(str, Enum):
    """The nature of the claim - what kind of knowledge does the hypothesis propose?"""
    DESCRIPTIVE = "descriptive"
    ASSOCIATIVE = "associative"
    CAUSAL = "causal"


class StructuralType(str, Enum):
    """Complexity of the relationship between variables"""
    SIMPLE = "simple"
    COMPLEX = "complex"


class PredictiveType(str, Enum):
    """Specificity of the predicted outcome"""
    NON_DIRECTIONAL = "non_directional"
    DIRECTIONAL = "directional"


class FunctionalType(str, Enum):
    """The role of the hypothesis in the investigation"""
    SCIENTIFIC = "scientific"
    STATISTICAL = "statistical"
    WORKING = "working"


class TemporalType(str, Enum):
    """Stage of inquiry where the hypothesis is used"""
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class SpecificHypothesisType(str, Enum):
    """Specific, well-defined hypothesis types"""
    COMPARATIVE_PERFORMANCE = "comparative_performance"
    TRANSFERABILITY = "transferability"
    IMPLEMENTATION = "implementation"
    OTHER = "other"


class HypothesisClassification(BaseModel):
    """
    A comprehensive classification of a research hypothesis along multiple axes based on
    the presentation by Andrey Ustyuzhanin
    https://gamma.app/docs/The-Architecture-of-Inquiry-A-Comprehensive-Taxonomy-of-Scientifi-mqdpe9i68rtf6os
    
    This model is designed for use with OpenAI structured outputs to automatically
    classify hypotheses from research papers.
    """
    
    # The original hypothesis text
    hypothesis_text: str = Field(
        ...,
        description="The brief text of the hypothesis in the paper"
    )
    
    # Epistemic Axis
    epistemic_type: EpistemicType = Field(
        ...,
        description=(
            "The nature of the claim: "
            "DESCRIPTIVE (Proposes the existence, frequency, or characteristics of a phenomenon.), "
            "ASSOCIATIVE (Proposes a systematic relationship between two or more variables. Correlation does not imply causation.), "
            "CAUSAL (The most ambitious claim; proposes that a change in one variable causes a change in another. Requires rigorous experimental research to establish.)"
        )
    )
    
    epistemic_justification: str = Field(
        ...,
        description="Brief explanation of why this epistemic classification was chosen"
    )
    
    # Structural Axis
    structural_type: StructuralType = Field(
        ...,
        description=(
            "Complexity of relationships: "
            "SIMPLE (Proposes a relationship between exactly two variables (one independent, one dependent).), "
            "COMPLEX (Involves more than two variables.)"
        )
    )
    
    variables_identified: List[str] = Field(
        default_factory=list,
        description="List of variables mentioned in the hypothesis"
    )
    
    # Predictive Axis
    predictive_type: PredictiveType = Field(
        ...,
        description=(
            "Specificity of outcome: "
            "NON_DIRECTIONAL (relationship exists but direction unspecified), "
            "DIRECTIONAL (specific direction of relationship predicted)"
        )
    )
    
    predicted_direction: Optional[str] = Field(
        None,
        description="If directional, describe the predicted direction (e.g., 'positive correlation', 'X increases Y')"
    )
    
    # Functional Axis
    functional_type: FunctionalType = Field(
        ...,
        description=(
            "Role in investigation: "
            "SCIENTIFIC (The substantive, conceptual statement the researcher is interested in.), "
            "STATISTICAL (The formal, mathematical translation of the scientific hypothesis used for testing. The null hypothesis (H₀) is a statement of no effect, which the researcher aims to reject.), "
            "WORKING (A provisional, flexible assumption that guides early, exploratory research.)"
        )
    )
    
    # Temporal Axis
    temporal_type: TemporalType = Field(
        ...,
        description=(
            "Stage of inquiry: "
            "EXPLORATORY (Conducted to explore a topic and generate novel hypotheses from the data.), "
            "CONFIRMATORY (Designed to rigorously test a specific, pre-stated hypothesis.)"
        )
    )
    
    # Specific Types
    specific_type: SpecificHypothesisType = Field(
        ...,
        description=(
            "Specific hypothesis type if applicable: "
            "COMPARATIVE_PERFORMANCE ('Method X better than Y for Z'), "
            "TRANSFERABILITY (Tests whether a known principle or method can be effectively transferred to a new context or problem.), "
            "IMPLEMENTATION (Tests the successful implementation of a design based on established scientific principles.), "
            "OTHER (doesn't fit specific categories)"
        )
    )
    
    specific_type_details: Optional[str] = Field(
        None,
        description="Additional details about the specific hypothesis type if not OTHER"
    )
    
    # Additional metadata
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in the classification (0-1), if applicable"
    )
    
    notes: Optional[str] = Field(
        None,
        description="Any additional notes or observations about the hypothesis"
    )
    
    class Config:
        use_enum_values = True


# Pydantic schema examples built from HypothesisClassification instances.
EXAMPLE_CLASSIFICATIONS = [
    HypothesisClassification(
        hypothesis_text="Students who eat breakfast will perform better on a math exam",
        epistemic_type=EpistemicType.CAUSAL,
        epistemic_justification="Implies eating breakfast causes better performance",
        structural_type=StructuralType.SIMPLE,
        variables_identified=["eating breakfast", "math exam performance"],
        predictive_type=PredictiveType.DIRECTIONAL,
        predicted_direction="Eating breakfast increases math exam performance",
        functional_type=FunctionalType.SCIENTIFIC,
        temporal_type=TemporalType.CONFIRMATORY,
        specific_type=SpecificHypothesisType.OTHER,
        specific_type_details=None,
        confidence_score=0.95,
        notes="Clear causal and directional hypothesis suitable for experimental testing",
    ),
    HypothesisClassification(
        hypothesis_text="42% of Europeans never exercise",
        epistemic_type=EpistemicType.DESCRIPTIVE,
        epistemic_justification="Describes the frequency of a phenomenon (exercise behavior) in a population",
        structural_type=StructuralType.SIMPLE,
        variables_identified=["Europeans", "exercise frequency"],
        predictive_type=PredictiveType.NON_DIRECTIONAL,
        predicted_direction=None,
        functional_type=FunctionalType.SCIENTIFIC,
        temporal_type=TemporalType.CONFIRMATORY,
        specific_type=SpecificHypothesisType.OTHER,
        specific_type_details=None,
        confidence_score=0.98,
        notes="Purely descriptive claim about population characteristics",
    ),
    HypothesisClassification(
        hypothesis_text="Smoking is a prominent cause of lung cancer",
        epistemic_type=EpistemicType.CAUSAL,
        epistemic_justification="Explicitly states a causal relationship between smoking and lung cancer",
        structural_type=StructuralType.SIMPLE,
        variables_identified=["smoking", "lung cancer"],
        predictive_type=PredictiveType.DIRECTIONAL,
        predicted_direction="Smoking increases risk of lung cancer",
        functional_type=FunctionalType.SCIENTIFIC,
        temporal_type=TemporalType.CONFIRMATORY,
        specific_type=SpecificHypothesisType.OTHER,
        specific_type_details=None,
        confidence_score=0.99,
        notes="Classic simple causal hypothesis with two variables",
    ),
    HypothesisClassification(
        hypothesis_text="High-sugar diet and sedentary activity levels are more likely to develop depression",
        epistemic_type=EpistemicType.CAUSAL,
        epistemic_justification="Proposes that diet and activity cause depression",
        structural_type=StructuralType.COMPLEX,
        variables_identified=["high-sugar diet", "sedentary activity levels", "depression"],
        predictive_type=PredictiveType.DIRECTIONAL,
        predicted_direction="High-sugar diet and sedentary lifestyle increase likelihood of depression",
        functional_type=FunctionalType.SCIENTIFIC,
        temporal_type=TemporalType.CONFIRMATORY,
        specific_type=SpecificHypothesisType.OTHER,
        specific_type_details=None,
        confidence_score=0.92,
        notes="Complex hypothesis with multiple independent variables affecting one dependent variable",
    ),
    HypothesisClassification(
        hypothesis_text="There is a difference in job satisfaction between those who receive regular feedback and those who do not",
        epistemic_type=EpistemicType.ASSOCIATIVE,
        epistemic_justification="States a relationship exists but does not claim causation",
        structural_type=StructuralType.SIMPLE,
        variables_identified=["regular feedback", "job satisfaction"],
        predictive_type=PredictiveType.NON_DIRECTIONAL,
        predicted_direction=None,
        functional_type=FunctionalType.SCIENTIFIC,
        temporal_type=TemporalType.CONFIRMATORY,
        specific_type=SpecificHypothesisType.OTHER,
        specific_type_details=None,
        confidence_score=0.94,
        notes="Non-directional hypothesis - predicts difference but not which direction",
    ),
    HypothesisClassification(
        hypothesis_text="Method X is better than method Y for problem Z",
        epistemic_type=EpistemicType.CAUSAL,
        epistemic_justification="Implies using Method X causes better outcomes than Method Y",
        structural_type=StructuralType.SIMPLE,
        variables_identified=["method choice (X vs Y)", "performance on problem Z"],
        predictive_type=PredictiveType.DIRECTIONAL,
        predicted_direction="Method X produces better results than Method Y",
        functional_type=FunctionalType.SCIENTIFIC,
        temporal_type=TemporalType.CONFIRMATORY,
        specific_type=SpecificHypothesisType.COMPARATIVE_PERFORMANCE,
        specific_type_details="Directly compares two methods on a specific problem",
        confidence_score=0.97,
        notes="Classic comparative performance claim - testable through direct comparison",
    ),
    HypothesisClassification(
        hypothesis_text="Method X can be transferred from problem Y to problem Z",
        epistemic_type=EpistemicType.CAUSAL,
        epistemic_justification="Proposes that applying Method X will produce results on the new problem Z",
        structural_type=StructuralType.SIMPLE,
        variables_identified=["method X application", "success on problem Z"],
        predictive_type=PredictiveType.DIRECTIONAL,
        predicted_direction="Method X will be effective on problem Z",
        functional_type=FunctionalType.SCIENTIFIC,
        temporal_type=TemporalType.CONFIRMATORY,
        specific_type=SpecificHypothesisType.TRANSFERABILITY,
        specific_type_details="Tests whether a known method generalizes to a new context",
        confidence_score=0.96,
        notes="Transferability hypothesis - key for establishing generalizability of methods",
    ),
    HypothesisClassification(
        hypothesis_text="A device built to these specs will achieve 45% thermal efficiency",
        epistemic_type=EpistemicType.CAUSAL,
        epistemic_justification="Claims that building according to specifications causes specific efficiency outcome",
        structural_type=StructuralType.SIMPLE,
        variables_identified=["device specifications", "thermal efficiency"],
        predictive_type=PredictiveType.DIRECTIONAL,
        predicted_direction="Device will achieve 45% thermal efficiency",
        functional_type=FunctionalType.SCIENTIFIC,
        temporal_type=TemporalType.CONFIRMATORY,
        specific_type=SpecificHypothesisType.IMPLEMENTATION,
        specific_type_details="Tests whether design based on established principles achieves predicted performance",
        confidence_score=0.93,
        notes="Implementation hypothesis - verifies engineering design based on scientific principles",
    ),
]

HypothesisClassification.Config.json_schema_extra = {
    "examples": [example.dict() for example in EXAMPLE_CLASSIFICATIONS]
}
