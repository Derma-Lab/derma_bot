import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node, print_tree
from openai import AzureOpenAI
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

class DermAgent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4'):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2023-03-15-preview"
        )
        self.messages = [{"role": "system", "content": instruction}]
        
        if examplers is not None:
            for exampler in examplers:
                self.messages.append({"role": "user", "content": exampler['question']})
                self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def chat(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.messages,
            temperature=0.7
        )
        response_text = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text

class DermGroup:
    def __init__(self, goal, members, question, examplers=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            agent = DermAgent(
                f"You are a {member_info['role']} who {member_info['expertise_description'].lower()}.",
                role=member_info['role']
            )
            agent.chat(f"You are a {member_info['role']} who {member_info['expertise_description'].lower()}.")
            self.members.append(agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            
            # Identify lead and assistant members
            for member in self.members:
                if 'lead' in member.role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)
            
            if lead_member is None:
                lead_member = assist_members[0]
            
            # Generate delivery prompt
            delivery_prompt = f'''You are the lead of the dermatology group which aims to {self.goal}. You have the following assistant dermatologists who work for you:'''
            for member in assist_members:
                delivery_prompt += f"\n{member.role}"
            
            delivery_prompt += f"\n\nGiven the dermatology case, what specific investigations are needed from each assistant?\nCase: {self.question}"
            
            # Get lead member's delivery
            delivery = lead_member.chat(delivery_prompt)
            
            # Gather investigations from assistants
            investigations = []
            for member in assist_members:
                investigation = member.chat(
                    f"You are in a dermatology group where the goal is to {self.goal}. "
                    f"Your group lead is requesting the following investigations:\n{delivery}\n\n"
                    f"Please provide your investigation summary focusing on your specific expertise."
                )
                investigations.append([member.role, investigation])
            
            # Compile investigations
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += f"[{investigation[0]}]\n{investigation[1]}\n"

            # Generate final response
            if self.examplers:
                response_prompt = (
                    f"Review the investigations from your assistant dermatologists:\n{gathered_investigation}\n\n"
                    f"Based on these example cases:\n{self.examplers}\n\n"
                    f"Provide your assessment of the case: {self.question}"
                )
            else:
                response_prompt = (
                    f"Review the investigations from your assistant dermatologists:\n{gathered_investigation}\n\n"
                    f"Provide your assessment of the case: {self.question}"
                )

            return lead_member.chat(response_prompt)

def parse_hierarchy(info, emojis):
    """Parse hierarchy of dermatology team structure"""
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node(f"{child} ({emojis[count]})", agent)
                    agents.append(child_agent)
        else:
            agent = Node(f"{expert} ({emojis[count]})", moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info: str) -> Dict:
    """Parse dermatology group information"""
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def determine_difficulty(case_info: str, difficulty: str = 'adaptive') -> str:
    """Determine complexity of dermatology case"""
    if difficulty != 'adaptive':
        return difficulty

    difficulty_prompt = f"""Assess the complexity of this dermatology case:
{case_info}

Choose from:
1) Basic: Can be handled by a single dermatologist
2) Intermediate: Requires consultation between multiple dermatology specialists
3) Advanced: Requires collaboration between multiple dermatology teams

Provide your assessment and reasoning."""

    assessor = DermAgent(
        instruction='You are a dermatology expert who assesses case complexity.',
        role='dermatology assessor'
    )
    
    response = assessor.chat(difficulty_prompt)
    
    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    else:
        return 'advanced'

def process_basic_query(case_info: str, examplers: List, model: str) -> Dict:
    """Process basic dermatology case"""
    derm = DermAgent(
        instruction='You are a comprehensive dermatologist who handles general dermatology cases.',
        role='dermatologist',
        examplers=examplers,
        model_info=model
    )
    
    assessment = derm.chat(
        f"Please provide comprehensive assessment for this dermatology case:\n{case_info}\n\n"
        "Include:\n"
        "1. Diagnosis\n"
        "2. Treatment plan\n"
        "3. Follow-up recommendations"
    )
    
    return {"assessment": assessment}

def process_intermediate_query(case_info: str, examplers: List, model: str) -> Dict:
    """Process intermediate dermatology case requiring multiple specialists"""
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    
    # Setup recruiter
    recruit_prompt = "You are an experienced dermatologist who recruits specialists for complex cases."
    recruiter = DermAgent(instruction=recruit_prompt, role='recruiter')
    
    # Get specialist recommendations
    specialist_prompt = f"""Case: {case_info}
You can recruit 3 dermatology specialists. What specialists would be most appropriate for this case?
Also specify their interaction hierarchy (e.g., Medical Derm > Surgical Derm) or if independent.

Format as:
1. Medical Dermatologist - Expertise in general skin conditions - Hierarchy: Independent
2. Surgical Dermatologist - Expertise in procedures - Hierarchy: Medical Derm > Surgical Derm
3. Dermatopathologist - Expertise in skin pathology - Hierarchy: Independent"""

    specialists = recruiter.chat(specialist_prompt)
    
    # Parse specialists and create hierarchy
    specialist_info = [s.split(" - Hierarchy: ") for s in specialists.split('\n') if s]
    specialist_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in specialist_info]
    
    emojis = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F']
    hierarchy = parse_hierarchy(specialist_data, emojis)

    # Create specialist agents and get opinions
    specialist_opinions = []
    for info in specialist_data:
        agent_role = info[0].split('-')[0].strip()
        expertise = info[0].split('-')[1].strip()
        
        specialist = DermAgent(
            f"You are a {agent_role} who {expertise.lower()}",
            role=agent_role,
            model_info=model
        )
        
        opinion = specialist.chat(f"Please provide your specialist assessment for:\n{case_info}")
        specialist_opinions.append((agent_role, opinion))
    
    # Get final assessment
    moderator = DermAgent(
        "You are a senior dermatologist who reviews specialist opinions and makes final decisions.",
        "Senior Dermatologist",
        model_info=model
    )
    
    opinions_text = "\n\n".join([f"{role}:\n{opinion}" for role, opinion in specialist_opinions])
    final_assessment = moderator.chat(
        f"Review these specialist opinions and provide final assessment:\n\n{opinions_text}\n\n"
        f"Case information:\n{case_info}"
    )
    
    return {
        "specialist_opinions": specialist_opinions,
        "final_assessment": final_assessment
    }

def process_advanced_query(case_info: str, model: str) -> Dict:
    """Process advanced dermatology case requiring multiple teams"""
    cprint("[INFO] Step 1. Team Formation", 'yellow', attrs=['blink'])
    
    # Setup team organizer
    organizer = DermAgent(
        "You are a senior dermatologist organizing multiple teams for complex cases.",
        "Team Organizer",
        model_info=model
    )
    
    # Get team recommendations
    team_prompt = f"""For this complex case: {case_info}
Organize 3 dermatology teams with different focuses. Each team should have 3 specialists.

Format as:
Group 1 - Medical Assessment Team
Member 1: Medical Dermatologist (Lead) - Expertise in complex medical dermatology
Member 2: Immunodermatologist - Expertise in autoimmune conditions
Member 3: Contact Dermatitis Specialist - Expertise in allergic reactions"""

    teams = organizer.chat(team_prompt)
    
    # Parse teams and create groups
    group_instances = []
    teams_info = [team.strip() for team in teams.split("Group") if team.strip()]
    
    for team_info in teams_info:
        parsed_team = parse_group_info(team_info)
        group = DermGroup(parsed_team['group_goal'], parsed_team['members'], case_info)
        group_instances.append(group)
    
    # Get team assessments
    team_assessments = []
    for group in group_instances:
        assessment = group.interact('internal')
        team_assessments.append((group.goal, assessment))
    
    # Get final recommendation
    final_reviewer = DermAgent(
        "You are a senior dermatologist who reviews team assessments and makes final recommendations.",
        "Senior Reviewer",
        model_info=model
    )
    
    assessments_text = "\n\n".join([f"{goal}:\n{assessment}" for goal, assessment in team_assessments])
    final_recommendation = final_reviewer.chat(
        f"Review all team assessments and provide final recommendation:\n\n{assessments_text}\n\n"
        f"Case information:\n{case_info}"
    )
    
    return {
        "team_assessments": team_assessments,
        "final_recommendation": final_recommendation
    }

def gather_patient_input() -> str:
    """Gather patient information from the end user"""
    cprint("[INFO] Please provide details about your dermatology case.", 'cyan')
    patient_info = input("Patient Info: ")
    return patient_info

def main():
    # Gather patient input
    patient_info = gather_patient_input()
    
    # Determine case complexity
    difficulty = determine_difficulty(patient_info)
    
    # Process the case based on complexity
    if difficulty == 'basic':
        result = process_basic_query(patient_info, examplers=None, model='gpt-4')
        cprint(f"[RESULT] Basic Assessment:\n{result['assessment']}", 'green')
    elif difficulty == 'intermediate':
        result = process_intermediate_query(patient_info, examplers=None, model='gpt-4')
        cprint(f"[RESULT] Intermediate Assessment:\n{result['final_assessment']}", 'green')
    elif difficulty == 'advanced':
        result = process_advanced_query(patient_info, model='gpt-4')
        cprint(f"[RESULT] Advanced Recommendation:\n{result['final_recommendation']}", 'green')

if __name__ == "__main__":
    main()