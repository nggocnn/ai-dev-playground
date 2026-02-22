import os
import json
import re
from typing import Dict, List, Optional, Any
from llama_cpp import Llama


class LlamaModel:
    """
    A wrapper class for LLaMA3 model operations with proper exception handling.
    
    This class provides a clean interface for loading and using LLaMA3 models
    locally for text generation tasks.
    """
    
    def __init__(self, model_path: str, n_ctx: int = 2048, verbose: bool = False):
        """
        Initialize the LLaMA3 model.
        
        Args:
            model_path (str): Path to the LLaMA3 model file (.gguf format)
            n_ctx (int): Context window size (default: 2048)
            verbose (bool): Enable verbose logging
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model fails to load
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.llm = None
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        try:
            print(f"Loading LLaMA model from {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                verbose=verbose,
                n_threads=4,  # Optimize for local execution
                n_gpu_layers=0  # CPU-only execution for compatibility
            )
            print(f"LLaMA model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load LLaMA3 model: {e}")
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, 
                     temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        """
        Generate text using the loaded LLaMA3 model.
        
        Args:
            prompt (str): Input prompt for text generation
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.0 = deterministic, 1.0 = creative)
            stop (List[str], optional): Stop sequences
            
        Returns:
            str: Generated text
            
        Raises:
            RuntimeError: If text generation fails
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Please initialize the model first.")
            
        try:
            if self.verbose:
                print(f"Generating text with prompt length: {len(prompt)} characters")
                
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or ["Human:", "User:", "\n\n---"],
                echo=False
            )
            
            generated_text = output['choices'][0]['text'].strip()
            
            if self.verbose:
                print(f"Generated {len(generated_text)} characters")
                
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate text: {e}")
    
    def __del__(self):
        """Cleanup resources when object is destroyed."""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm


class ResumeGenerator:
    """
    Main class for generating professional resumes using LLaMA3.
    
    This class handles loading prompts, processing user data, and generating
    formatted resumes in Markdown format.
    """
    
    def __init__(self, model_path: str, prompts_file: str = "prompts.txt"):
        """
        Initialize the Resume Generator.
        
        Args:
            model_path (str): Path to the LLaMA model file
            prompts_file (str): Path to the prompts file
        """
        self.model_path = model_path
        self.prompts_file = prompts_file
        self.prompts = {}
        self.llama_model = None
        
        # Load prompts
        self.load_prompts()
        
    def load_prompts(self) -> None:
        """Load resume generation prompts from file."""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse prompts by splitting on prompt names
            prompt_sections = re.split(r'^([A-Z_]+_PROMPT):', content, flags=re.MULTILINE)
            
            for i in range(1, len(prompt_sections), 2):
                prompt_name = prompt_sections[i].strip()
                prompt_content = prompt_sections[i + 1].strip()
                self.prompts[prompt_name] = prompt_content
                
            print(f"Loaded {len(self.prompts)} prompts from {self.prompts_file}")
            
        except FileNotFoundError:
            print(f"Prompts file not found: {self.prompts_file}")
            # Fallback to default prompt
            self.prompts["RESUME_GENERATION_PROMPT"] = self._get_default_prompt()
        except Exception as e:
            print(f"Error loading prompts: {e}")
            self.prompts["RESUME_GENERATION_PROMPT"] = self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Return a default resume generation prompt."""
        return """You are a professional resume writer. Generate a well-structured, professional resume in Markdown format based on the provided user information.

Follow these guidelines:
1. Use proper Markdown formatting with headers, bullet points, and emphasis
2. Create sections: Header/Contact, Professional Summary, Work Experience, Education, Skills
3. Make the content tailored to the user's experience level and industry
4. Use action verbs and quantifiable achievements where possible
5. Keep the tone professional and concise

User Information:
{user_data}

Please generate a complete professional resume in Markdown format. Start with # [Full Name] as the main header."""
    
    def initialize_model(self) -> bool:
        """
        Initialize the LLaMA model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.llama_model = LlamaModel(self.model_path, n_ctx=4096)
            return True
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Model initialization failed: {e}")
            return False
    
    def format_user_data(self, user_data: Dict[str, Any]) -> str:
        """
        Format user data into a readable string for the LLM.
        
        Args:
            user_data (dict): User information dictionary
            
        Returns:
            str: Formatted user data string
        """
        formatted_parts = []
        
        # Personal Information
        if 'personal_info' in user_data:
            personal = user_data['personal_info']
            formatted_parts.append("=== PERSONAL INFORMATION ===")
            for key, value in personal.items():
                formatted_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        # Summary/Objective
        if 'summary' in user_data:
            formatted_parts.append("\n=== PROFESSIONAL SUMMARY ===")
            formatted_parts.append(user_data['summary'])
        elif 'objective' in user_data:
            formatted_parts.append("\n=== CAREER OBJECTIVE ===")
            formatted_parts.append(user_data['objective'])
        
        # Experience
        if 'experience' in user_data:
            formatted_parts.append("\n=== WORK EXPERIENCE ===")
            for exp in user_data['experience']:
                formatted_parts.append(f"\nPosition: {exp.get('title', 'N/A')}")
                formatted_parts.append(f"Company: {exp.get('company', 'N/A')}")
                formatted_parts.append(f"Location: {exp.get('location', 'N/A')}")
                formatted_parts.append(f"Duration: {exp.get('duration', 'N/A')}")
                if 'responsibilities' in exp:
                    formatted_parts.append("Responsibilities:")
                    for resp in exp['responsibilities']:
                        formatted_parts.append(f"- {resp}")
        
        # Education
        if 'education' in user_data:
            formatted_parts.append("\n=== EDUCATION ===")
            for edu in user_data['education']:
                formatted_parts.append(f"\nDegree: {edu.get('degree', 'N/A')}")
                formatted_parts.append(f"School: {edu.get('school', 'N/A')}")
                formatted_parts.append(f"Location: {edu.get('location', 'N/A')}")
                formatted_parts.append(f"Graduation: {edu.get('graduation', 'N/A')}")
                if 'gpa' in edu:
                    formatted_parts.append(f"GPA: {edu['gpa']}")
                if 'relevant_coursework' in edu:
                    formatted_parts.append(f"Relevant Coursework: {', '.join(edu['relevant_coursework'])}")
        
        # Skills
        if 'skills' in user_data:
            formatted_parts.append("\n=== SKILLS ===")
            skills = user_data['skills']
            for category, skill_list in skills.items():
                if isinstance(skill_list, list):
                    formatted_parts.append(f"{category.replace('_', ' ').title()}: {', '.join(skill_list)}")
                else:
                    formatted_parts.append(f"{category.replace('_', ' ').title()}: {skill_list}")
        
        # Projects
        if 'projects' in user_data:
            formatted_parts.append("\n=== PROJECTS ===")
            for project in user_data['projects']:
                formatted_parts.append(f"\nProject: {project.get('name', 'N/A')}")
                formatted_parts.append(f"Description: {project.get('description', 'N/A')}")
                if 'technologies' in project:
                    if isinstance(project['technologies'], list):
                        formatted_parts.append(f"Technologies: {', '.join(project['technologies'])}")
                    else:
                        formatted_parts.append(f"Technologies: {project['technologies']}")
                if 'github' in project:
                    formatted_parts.append(f"GitHub: {project['github']}")
        
        # Additional sections
        for section in ['certifications', 'achievements', 'activities']:
            if section in user_data:
                formatted_parts.append(f"\n=== {section.upper()} ===")
                items = user_data[section]
                if isinstance(items, list):
                    for item in items:
                        formatted_parts.append(f"- {item}")
                else:
                    formatted_parts.append(str(items))
        
        return "\n".join(formatted_parts)
    
    def select_prompt(self, user_type: str) -> str:
        """
        Select appropriate prompt based on user type.
        
        Args:
            user_type (str): Type of user ('technical', 'entry_level', 'general')
            
        Returns:
            str: Selected prompt template
        """
        prompt_mapping = {
            'technical': 'TECHNICAL_RESUME_PROMPT',
            'entry_level': 'ENTRY_LEVEL_RESUME_PROMPT',
            'general': 'RESUME_GENERATION_PROMPT'
        }
        
        prompt_key = prompt_mapping.get(user_type, 'RESUME_GENERATION_PROMPT')
        return self.prompts.get(prompt_key, self.prompts.get('RESUME_GENERATION_PROMPT', self._get_default_prompt()))
    
    def generate_resume(self, user_data: Dict[str, Any]) -> str:
        """
        Generate a resume for the given user data.
        
        Args:
            user_data (dict): User information dictionary
            
        Returns:
            str: Generated resume in Markdown format
            
        Raises:
            RuntimeError: If model is not initialized or generation fails
        """
        if self.llama_model is None:
            raise RuntimeError("LLaMA model not initialized. Call initialize_model() first.")
        
        # Format user data
        formatted_data = self.format_user_data(user_data)
        
        # Select appropriate prompt
        user_type = user_data.get('type', 'general')
        prompt_template = self.select_prompt(user_type)
        
        # Create final prompt
        final_prompt = prompt_template.format(user_data=formatted_data)
        
        print(f"Generating resume for {user_data.get('personal_info', {}).get('full_name', 'User')}...")
        
        # Generate resume
        try:
            resume_content = self.llama_model.generate_text(
                prompt=final_prompt,
                max_tokens=2000,
                temperature=0.7,
                stop=["---", "Human:", "User:"]
            )
            
            print("Resume generated successfully!")
            return resume_content
            
        except Exception as e:
            raise RuntimeError(f"Resume generation failed: {e}")
    
    def save_resume(self, resume_content: str, filename: str, output_dir: str = "outputs") -> str:
        """
        Save resume content to a Markdown file.
        
        Args:
            resume_content (str): Generated resume content
            filename (str): Output filename (without extension)
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure filename ends with .md
        if not filename.endswith('.md'):
            filename += '.md'
        
        output_path = os.path.join(output_dir, filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(resume_content)
            
            print(f"Resume saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Failed to save resume: {e}")
            return ""


def download_model_if_needed(model_path: str) -> bool:
    """
    Download LLaMA3 model if it doesn't exist locally.
    
    Args:
        model_path (str): Path where model should be stored
        
    Returns:
        bool: True if model is available, False otherwise
    """
    if os.path.exists(model_path):
        print(f"Model found at: {model_path}")
        return True
    
    print(f"Model not found at: {model_path}")
    print("\nTo run this script, you need to download a LLaMA3 model.")
    print("Recommended models:")
    print("   1. Llama-3-8B-Instruct (4-bit quantized): ~5GB")
    print("   2. Llama-3-7B-Instruct (4-bit quantized): ~4GB")
    print("   3. Llama-3-70B-Instruct (4-bit quantized): ~40GB")
    
    print(f"\nDownload example (using wget):")
    print(f"   wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -O {model_path}")
    
    return False


def main():
    """Main function to run the resume generation system."""
    print("Resume Generation Using LLaMA3 Locally")
    print("=" * 50)
    
    # Configuration
    model_path = "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    input_file = "input.json"
    prompts_file = "prompts.txt"
    output_dir = "outputs"
    
    # Check if model exists
    if not download_model_if_needed(model_path):
        print("Cannot proceed without a model. Please download a model first.")
        return
    
    # Load user data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sample_users = data.get('sample_users', [])
        print(f"Loaded {len(sample_users)} user profiles from {input_file}")
    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {input_file}: {e}")
        return
    
    # Initialize resume generator
    try:
        generator = ResumeGenerator(model_path, prompts_file)
        
        # Initialize model
        if not generator.initialize_model():
            print("Failed to initialize model. Exiting.")
            return
            
    except Exception as e:
        print(f"Failed to initialize resume generator: {e}")
        return
    
    # Generate resumes for all users
    print(f"\nGenerating resumes for {len(sample_users)} users...")
    print("-" * 50)
    
    successful_generations = 0
    
    for i, user_data in enumerate(sample_users, 1):
        try:
            user_name = user_data.get('personal_info', {}).get('full_name', f'User_{i}')
            print(f"\nProcessing User {i}: {user_name}")
            
            # Generate resume
            resume_content = generator.generate_resume(user_data)
            
            # Save resume
            filename = f"resume_{user_name.lower().replace(' ', '_')}"
            output_path = generator.save_resume(resume_content, filename, output_dir)
            
            if output_path:
                successful_generations += 1
                print(f"Resume {i} completed: {output_path}")
            
        except Exception as e:
            print(f"Failed to generate resume for user {i}: {e}")
            continue
    
    # Summary
    print(f"\nResume generation complete!")
    print(f"Successfully generated: {successful_generations}/{len(sample_users)} resumes")
    print(f"Output directory: {output_dir}")
    
    if successful_generations > 0:
        print(f"\nSample resumes generated:")
        for file in os.listdir(output_dir):
            if file.endswith('.md'):
                print(f"   - {os.path.join(output_dir, file)}")


if __name__ == "__main__":
    main()
