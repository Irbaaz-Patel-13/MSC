"""
In-Context Affordance Reasoning Module
=======================================
Implements the three-step VLM-based affordance reasoning pipeline from AffordGrasp:
  Step 1: Task Analysis - Extract task goal from implicit user instructions
  Step 2: Object Identification - Identify the most task-relevant object in the scene
  Step 3: Part & Affordance Reasoning - Determine optimal graspable part based on affordances

Uses GPT-4o as the backbone VLM, matching the paper's implementation.

Reference: AffordGrasp, Section III-A: In-Context Affordance Reasoning
"""

import os
import json
import base64
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("[Warning] openai package not installed. VLM reasoning will use mock mode.")
    OpenAI = None

from config import VLMConfig


@dataclass
class TaskAnalysisResult:
    """Output of Step 1: Task Analysis."""
    task_goal: str
    functional_requirements: list
    required_object_type: str
    required_properties: list
    raw_response: dict


@dataclass
class ObjectIdentificationResult:
    """Output of Step 2: Relevant Object Identification."""
    target_object: str
    object_description: str
    confidence: float
    approximate_location: str
    raw_response: dict


@dataclass
class AffordanceReasoningResult:
    """Output of Step 3: Part and Affordance Reasoning."""
    object_parts: list
    optimal_grasp_part: str
    grasp_reasoning: str
    grasp_approach: str
    raw_response: dict


@dataclass
class FullReasoningResult:
    """Combined output from all three reasoning steps."""
    task_analysis: TaskAnalysisResult
    object_identification: ObjectIdentificationResult
    affordance_reasoning: AffordanceReasoningResult
    user_instruction: str


class AffordanceReasoner:
    """
    Three-step in-context affordance reasoning using GPT-4o.
    
    This is the core intelligence module of AffordGrasp. Given a user's
    natural language instruction and an RGB image of the scene, it:
    1. Extracts the implicit task goal
    2. Identifies which object in the cluttered scene is most relevant
    3. Reasons about object parts and affordances to select the optimal grasp region
    
    Usage:
        config = VLMConfig(api_key="sk-...")
        reasoner = AffordanceReasoner(config)
        result = reasoner.reason(
            instruction="I want to drink some coffee",
            scene_image=rgb_array
        )
        print(result.affordance_reasoning.optimal_grasp_part)  # "handle"
    """
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.client = None
        self.mock_mode = False
        
        if OpenAI is not None and config.api_key:
            self.client = OpenAI(api_key=config.api_key)
        else:
            print("[AffordanceReasoner] Running in MOCK mode (no API key).")
            print("  Set OPENAI_API_KEY environment variable for real VLM reasoning.")
            self.mock_mode = True
    
    def reason(
        self,
        instruction: str,
        scene_image: np.ndarray,
        verbose: bool = True
    ) -> FullReasoningResult:
        """
        Execute the full three-step affordance reasoning pipeline.
        
        Args:
            instruction: User's natural language instruction (e.g., "I want to scoop something")
            scene_image: RGB image of the scene as numpy array (H, W, 3)
            verbose: Print reasoning steps
        
        Returns:
            FullReasoningResult containing all three steps' outputs
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"[AffordanceReasoner] Starting reasoning pipeline")
            print(f"  Instruction: \"{instruction}\"")
            print(f"{'='*60}")
        
        # Step 1: Task Analysis
        if verbose:
            print("\n--- Step 1: Task Analysis ---")
        task_result = self._step1_task_analysis(instruction)
        if verbose:
            print(f"  Task Goal: {task_result.task_goal}")
            print(f"  Required Object: {task_result.required_object_type}")
            print(f"  Requirements: {task_result.functional_requirements}")
        
        # Step 2: Object Identification (uses image)
        if verbose:
            print("\n--- Step 2: Object Identification ---")
        obj_result = self._step2_object_identification(
            scene_image, task_result
        )
        if verbose:
            print(f"  Target Object: {obj_result.target_object}")
            print(f"  Confidence: {obj_result.confidence:.2f}")
            print(f"  Description: {obj_result.object_description}")
        
        # Step 3: Part & Affordance Reasoning
        if verbose:
            print("\n--- Step 3: Part & Affordance Reasoning ---")
        affordance_result = self._step3_affordance_reasoning(
            task_result, obj_result
        )
        if verbose:
            print(f"  Parts found: {[p['part_name'] for p in affordance_result.object_parts]}")
            print(f"  Optimal Grasp Part: {affordance_result.optimal_grasp_part}")
            print(f"  Reasoning: {affordance_result.grasp_reasoning}")
            print(f"  Approach: {affordance_result.grasp_approach}")
        
        return FullReasoningResult(
            task_analysis=task_result,
            object_identification=obj_result,
            affordance_reasoning=affordance_result,
            user_instruction=instruction
        )
    
    # =========================================================================
    #  STEP 1: Task Analysis
    # =========================================================================
    
    def _step1_task_analysis(self, instruction: str) -> TaskAnalysisResult:
        """
        Extract the implicit task goal from the user's instruction.
        
        Example:
            "I want to scoop something" -> task_goal: "Use a concave tool to gather material"
        """
        if self.mock_mode:
            return self._mock_task_analysis(instruction)
        
        prompt = self.config.step1_task_analysis_template.format(
            instruction=instruction
        )
        
        response = self._call_vlm(prompt, image=None)
        data = self._parse_json_response(response)
        
        return TaskAnalysisResult(
            task_goal=data.get("task_goal", ""),
            functional_requirements=data.get("functional_requirements", []),
            required_object_type=data.get("required_object_type", ""),
            required_properties=data.get("required_properties", []),
            raw_response=data
        )
    
    # =========================================================================
    #  STEP 2: Relevant Object Identification
    # =========================================================================
    
    def _step2_object_identification(
        self,
        scene_image: np.ndarray,
        task_result: TaskAnalysisResult
    ) -> ObjectIdentificationResult:
        """
        Identify the most suitable object in the scene for the given task.
        Uses the scene image + task analysis to ground the object visually.
        """
        if self.mock_mode:
            return self._mock_object_identification(task_result)
        
        prompt = self.config.step2_object_identification_template.format(
            task_goal=task_result.task_goal,
            functional_requirements=", ".join(task_result.functional_requirements),
            required_object_type=task_result.required_object_type
        )
        
        response = self._call_vlm(prompt, image=scene_image)
        data = self._parse_json_response(response)
        
        return ObjectIdentificationResult(
            target_object=data.get("target_object", ""),
            object_description=data.get("object_description", ""),
            confidence=float(data.get("confidence", 0.0)),
            approximate_location=data.get("approximate_location", ""),
            raw_response=data
        )
    
    # =========================================================================
    #  STEP 3: Part & Affordance Reasoning
    # =========================================================================
    
    def _step3_affordance_reasoning(
        self,
        task_result: TaskAnalysisResult,
        obj_result: ObjectIdentificationResult
    ) -> AffordanceReasoningResult:
        """
        Decompose the target object into functional parts and reason about
        which part provides the optimal affordance for the task.
        
        Example:
            Object: wooden spoon, Task: scooping
            -> Parts: [bowl (functional), handle (graspable)]
            -> Optimal: handle (allows secure grip while bowl faces down for scooping)
        """
        if self.mock_mode:
            return self._mock_affordance_reasoning(task_result, obj_result)
        
        prompt = self.config.step3_affordance_reasoning_template.format(
            target_object=obj_result.target_object,
            task_goal=task_result.task_goal,
            functional_requirements=", ".join(task_result.functional_requirements)
        )
        
        response = self._call_vlm(prompt, image=None)
        data = self._parse_json_response(response)
        
        return AffordanceReasoningResult(
            object_parts=data.get("object_parts", []),
            optimal_grasp_part=data.get("optimal_grasp_part", ""),
            grasp_reasoning=data.get("grasp_reasoning", ""),
            grasp_approach=data.get("grasp_approach", "top"),
            raw_response=data
        )
    
    # =========================================================================
    #  VLM API CALLS
    # =========================================================================
    
    def _call_vlm(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        """Make a call to GPT-4o with optional image input."""
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        
        if image is not None:
            # Encode image as base64
            image_b64 = self._encode_image(image)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return response.choices[0].message.content
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode a numpy RGB image to base64 PNG string."""
        from PIL import Image
        import io
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from VLM response, handling markdown code blocks."""
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            print(f"[Warning] Failed to parse VLM JSON response: {e}")
            print(f"  Raw response: {response[:200]}...")
            return {}
    
    # =========================================================================
    #  MOCK RESPONSES (for testing without API key)
    # =========================================================================
    
    def _mock_task_analysis(self, instruction: str) -> TaskAnalysisResult:
        """Generate plausible mock task analysis based on instruction keywords."""
        instruction_lower = instruction.lower()
        
        # Mapping of common instructions to task analyses
        mock_db = {
            "drink": {
                "task_goal": "Drink a beverage",
                "functional_requirements": ["container for liquid", "handle for secure grip"],
                "required_object_type": "drinking vessel (cup/mug/glass)",
                "required_properties": ["has handle", "can hold liquid"]
            },
            "scoop": {
                "task_goal": "Scoop and transfer material",
                "functional_requirements": ["concave surface", "long handle for reach"],
                "required_object_type": "spoon or ladle",
                "required_properties": ["has bowl end", "has graspable handle"]
            },
            "hammer": {
                "task_goal": "Drive nails into a surface",
                "functional_requirements": ["heavy striking head", "graspable handle"],
                "required_object_type": "hammer",
                "required_properties": ["weighted head", "long handle"]
            },
            "cut": {
                "task_goal": "Cut or slice objects",
                "functional_requirements": ["sharp blade", "safe handle"],
                "required_object_type": "knife or scissors",
                "required_properties": ["has cutting edge", "has handle away from blade"]
            },
            "tighten": {
                "task_goal": "Tighten screws",
                "functional_requirements": ["flat or cross-head tip", "torque-friendly handle"],
                "required_object_type": "screwdriver",
                "required_properties": ["has tip for screw head", "has cylindrical handle"]
            },
            "cook": {
                "task_goal": "Cook food using cookware",
                "functional_requirements": ["heat-resistant surface", "handle for control"],
                "required_object_type": "pan or pot",
                "required_properties": ["flat cooking surface", "heat-safe handle"]
            },
            "flip": {
                "task_goal": "Flip food while cooking",
                "functional_requirements": ["flat thin blade", "long handle"],
                "required_object_type": "spatula",
                "required_properties": ["thin flat end", "graspable handle"]
            },
        }
        
        # Find best match
        best_match = None
        for key, analysis in mock_db.items():
            if key in instruction_lower:
                best_match = analysis
                break
        
        if best_match is None:
            best_match = {
                "task_goal": f"Complete the task described: {instruction}",
                "functional_requirements": ["appropriate tool for the task"],
                "required_object_type": "task-appropriate object",
                "required_properties": ["functional for the described task"]
            }
        
        return TaskAnalysisResult(
            task_goal=best_match["task_goal"],
            functional_requirements=best_match["functional_requirements"],
            required_object_type=best_match["required_object_type"],
            required_properties=best_match["required_properties"],
            raw_response=best_match
        )
    
    def _mock_object_identification(
        self, task_result: TaskAnalysisResult
    ) -> ObjectIdentificationResult:
        """Generate mock object identification."""
        obj_type = task_result.required_object_type
        
        return ObjectIdentificationResult(
            target_object=obj_type.split("(")[0].strip().split("/")[0].strip(),
            object_description=f"A {obj_type} suitable for {task_result.task_goal.lower()}",
            confidence=0.85,
            approximate_location="center-left of scene",
            raw_response={"mock": True}
        )
    
    def _mock_affordance_reasoning(
        self,
        task_result: TaskAnalysisResult,
        obj_result: ObjectIdentificationResult
    ) -> AffordanceReasoningResult:
        """Generate mock affordance reasoning with common object part decompositions."""
        target = obj_result.target_object.lower()
        
        # Common affordance knowledge base
        affordance_db = {
            "cup": {
                "parts": [
                    {"part_name": "rim", "function": "drinking edge", "graspable": False,
                     "affordance_for_task": "Used for drinking, must remain accessible"},
                    {"part_name": "body", "function": "holds liquid", "graspable": True,
                     "affordance_for_task": "Can be grasped but may be hot"},
                    {"part_name": "handle", "function": "grip point", "graspable": True,
                     "affordance_for_task": "Optimal grip for safe handling while drinking"}
                ],
                "optimal": "handle", "approach": "side"
            },
            "mug": {
                "parts": [
                    {"part_name": "rim", "function": "drinking edge", "graspable": False,
                     "affordance_for_task": "Drinking interface"},
                    {"part_name": "body", "function": "holds liquid", "graspable": True,
                     "affordance_for_task": "Alternative grip"},
                    {"part_name": "handle", "function": "primary grip", "graspable": True,
                     "affordance_for_task": "Secure grip for drinking"}
                ],
                "optimal": "handle", "approach": "side"
            },
            "spoon": {
                "parts": [
                    {"part_name": "bowl", "function": "scooping surface", "graspable": False,
                     "affordance_for_task": "Functional end for scooping"},
                    {"part_name": "handle", "function": "grip point", "graspable": True,
                     "affordance_for_task": "Long cylindrical grip for control"}
                ],
                "optimal": "handle", "approach": "top"
            },
            "hammer": {
                "parts": [
                    {"part_name": "head", "function": "striking surface", "graspable": False,
                     "affordance_for_task": "Impact surface for driving nails"},
                    {"part_name": "handle", "function": "grip and leverage", "graspable": True,
                     "affordance_for_task": "Provides leverage and control for striking"}
                ],
                "optimal": "handle", "approach": "side"
            },
            "knife": {
                "parts": [
                    {"part_name": "blade", "function": "cutting edge", "graspable": False,
                     "affordance_for_task": "Sharp edge for cutting - dangerous to grasp"},
                    {"part_name": "handle", "function": "safe grip", "graspable": True,
                     "affordance_for_task": "Ergonomic grip away from blade"}
                ],
                "optimal": "handle", "approach": "side"
            },
            "screwdriver": {
                "parts": [
                    {"part_name": "tip", "function": "screw engagement", "graspable": False,
                     "affordance_for_task": "Interfaces with screw head"},
                    {"part_name": "shaft", "function": "torque transfer", "graspable": False,
                     "affordance_for_task": "Transfers rotation to tip"},
                    {"part_name": "handle", "function": "grip for torque", "graspable": True,
                     "affordance_for_task": "Cylindrical grip allows torque application"}
                ],
                "optimal": "handle", "approach": "side"
            },
            "pan": {
                "parts": [
                    {"part_name": "cooking surface", "function": "heat transfer", "graspable": False,
                     "affordance_for_task": "Hot surface for cooking"},
                    {"part_name": "handle", "function": "control grip", "graspable": True,
                     "affordance_for_task": "Heat-insulated grip for pan control"}
                ],
                "optimal": "handle", "approach": "side"
            },
            "spatula": {
                "parts": [
                    {"part_name": "blade", "function": "flat flipping surface", "graspable": False,
                     "affordance_for_task": "Slides under food for flipping"},
                    {"part_name": "handle", "function": "control grip", "graspable": True,
                     "affordance_for_task": "Grip for precise flipping control"}
                ],
                "optimal": "handle", "approach": "top"
            },
        }
        
        # Find matching affordance info
        info = None
        for key in affordance_db:
            if key in target:
                info = affordance_db[key]
                break
        
        if info is None:
            info = {
                "parts": [
                    {"part_name": "body", "function": "main structure", "graspable": True,
                     "affordance_for_task": "Primary graspable region"}
                ],
                "optimal": "body", "approach": "top"
            }
        
        return AffordanceReasoningResult(
            object_parts=info["parts"],
            optimal_grasp_part=info["optimal"],
            grasp_reasoning=f"The {info['optimal']} provides the best grip for "
                          f"{task_result.task_goal.lower()}, keeping the functional "
                          f"parts of the object accessible for task execution.",
            grasp_approach=info["approach"],
            raw_response=info
        )
