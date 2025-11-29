import requests
from typing import Optional, Dict, List, Any
from smolagents import Tool # Import Tool class

import requests


class ClinicalTrialsSearchTool(Tool):
    name = "ClinicalTrialsSearchTool"
    description =""" Clinical trials analyst.
    Search ClinicalTrials.gov for trials related to a given disease / condition.

      Args:
            query_cond: Disease or condition (e.g., 'lung cancer', 'diabetes')
            query_term : Other terms (e.g., 'AREA[LastUpdatePostDate]RANGE[2023-01-15,MAX]').
            query_lead : Searches the LeadSponsorName
            max_results: Number of trials to return (max: 5000)
        
        Returns:
            List of strings, each string being a structured representation of a trial.

    Here is an overview of the 42 keys that can be used to extract selective information:
    # BASIC IDENTIFICATION 
        'nct_id'
        'brief_title'
        'official_title'
        'overall_status'
    
    # ORGANIZATION & SPONSOR 
        'lead_sponsor_name'
        'lead_sponsor_class'
        'collaborator_names' -> list
        'collaborator_classes' -> list
        'org_full_name'
    
    # KEY PERSONNEL & CONTACTS
        'overall_official_names' -> list
        'overall_official_affiliations' -> list
        'overall_official_roles' -> list
        'responsible_party_investigator_full_name'
        'responsible_party_investigator_affiliation'
        'num_collaborators'
    
    # SCIENTIFIC FOCUS AREAS 
        'conditions' -> list
        'intervention_names' -> list
        'intervention_types' -> list
        'phases' -> list
        'primary_outcome_measures' -> list
    
    # STUDY SCOPE & CAPACITY 
        'enrollment_count'
        'study_type'
        'num_locations'
        'location_facilities'  -> list
        'location_cities' -> list
        'location_states' -> list
        'location_countries' -> list
    
    # EXPERIENCE & TRACK RECORD 
        'study_first_post_date'
        'completion_date'
        'has_results'
        'num_collaborators_plus_lead'
    
    # THERAPEUTIC AREA EXPERTISE 
        'condition_mesh_terms' -> list
        'intervention_mesh_terms' -> list
        'primary_purpose'
    
    # CURRENT ACTIVITY STATUS 
        'last_update_post_date'
        'recruitment_status'
        'start_date'
        'primary_completion_date'
    
    # SECONDARY USEFUL FIELDS
        'oversight_has_dmc'
        'is_fda_regulated_drug'
        'is_fda_regulated_device'
        'location_statuses'
    
    # ADDITIONAL FIELDS to plot map or to lookup fo reference
        'citations' -> list
        'pmids' -> list
        'geopoints' -> list of dicts with lat/lon

    VERY IMPORTANT NOTE : When using this tools, avoid printing large datasets. Instead, process the data directly and only print summaries or final results.

    GOOD: trials = ClinicalTrialsSearchTool(...); process data; print(summary)
    BAD: trials = ClinicalTrialsSearchTool(...); print(trials)

    Now i will provide you with 3 exemple usage:

    exemple 1:
    task_description = "Find up to 10 trials for diabetes sponsored by Novo Nordisk, focusing on phase 4 studies."

    tool_arguments = {
        "query_cond": "diabetes",
        "query_term": "phase 4",
        "query_lead": "Novo Nordisk",
        "max_results": 10
    }

    clinical_trials_data = clinical_agent(
        task=task_description, 
        additional_args=tool_arguments
    )

    sponsors = []                                                                                                                                                                                        
    for trial in clinical_trials_data:                                                                                                                                                                         
        sponsors.extend(trial['lead_sponsor_name'])  

    print("Lead Sponsors for Retrieved Trials:",sponsors) == GOOD
    print("Clinical Trials Data:", clinical_trials_data) == BAD

    
    exemple 2:
    task_description = "Retrieve up to 5 clinical trials related to Alzheimer's disease conducted in Europe."

    tool_arguments = {
        "query_cond": "Alzheimer",
        "query_term": "Europe",
        "query_lead": "",
        "max_results": 5
    }

    clinical_trials_data = clinical_agent(
        task=task_description,
        additional_args=tool_arguments
    )

    locations = []
    for trial in clinical_trials_data:
        locations.extend(trial["facility_cities"])

    print("Cities of Retrieved Trials:", locations)    # GOOD
    print("Clinical Trials Data:", clinical_trials_data)  # BAD

    
    exemple 3:
    task_description = "Find up to 8 recruiting trials for breast cancer led by academic institutions."

    tool_arguments = {
        "query_cond": "breast cancer",
        "query_term": "recruiting",
        "query_lead": "university",
        "max_results": 8
    }

    clinical_trials_data = clinical_agent(
        task=task_description,
        additional_args=tool_arguments
    )

    statuses = []
    for trial in clinical_trials_data:
        statuses.append(trial["overall_status"])

    print("Trial Statuses:", statuses)    # GOOD
    # print("Clinical Trials Data:", clinical_trials_data)  # BAD


    """
    # The types for both inputs and output_type should be Pydantic formats, they can be either of these: 
    # ["string", "boolean","integer", "number", "image", "audio", "array", "object", "any", "null"].
    inputs = {
        "query_cond": {
            "type": "string",
            "description": "Searches for a disease or condition (e.g., 'lung cancer', 'diabete type 1').",
        "nullable": True
        },
        "query_term": {
        "type": "string",
        "description": (
            "Searches for Other terms "
            "Examples: AREA[LastUpdatePostDate]RANGE[2023-01-15,MAX]."
        ),
        "nullable": True
        },
        "query_lead": {
        "type": "string",
        "description": (
            "Searches the Lead Sponsor Name"
        ),
        "nullable": True
        },
        "max_results": {
            "type": "number",
            "description": "Number of trials to return (max: 100).",
            "default": 50,
            "nullable":True
        }
    }
    # We still return an array of strings, but each string is the YAML-like block for one trial.
    output_type = "array"


    def _extract_partner_info(self, api_response):
        """
        Extract core partner identification information from ClinicalTrials.gov API response.
        
        Args:
            api_response (dict): Raw API response from ClinicalTrials.gov
            
        Returns:
            dict: Flattened dictionary with 41 core fields plus citations and geopoints
        """
        
        # Helper function to safely navigate nested dicts
        def safe_get(data, *keys, default=None):
            for key in keys:
                if isinstance(data, dict):
                    data = data.get(key, {})
                else:
                    return default
            return data if data != {} else default
        
        protocol = api_response.get('protocolSection', {})
        
        # Extract basic identification
        identification = protocol.get('identificationModule', {})
        nct_id = identification.get('nctId')
        brief_title = identification.get('briefTitle')
        official_title = identification.get('officialTitle')
        org_full_name = safe_get(identification, 'organization', 'fullName')
        
        # Extract status information
        status = protocol.get('statusModule', {})
        overall_status = status.get('overallStatus')
        last_update_post_date = safe_get(status, 'lastUpdatePostDateStruct', 'date')
        recruitment_status = overall_status  # Same as overall status at protocol level
        start_date = safe_get(status, 'startDateStruct', 'date')
        primary_completion_date = safe_get(status, 'primaryCompletionDateStruct', 'date')
        completion_date = safe_get(status, 'completionDateStruct', 'date')
        study_first_post_date = safe_get(status, 'studyFirstPostDateStruct', 'date')
        
        # Extract sponsor/collaborator information
        sponsors = protocol.get('sponsorCollaboratorsModule', {})
        lead_sponsor = sponsors.get('leadSponsor', {})
        lead_sponsor_name = lead_sponsor.get('name')
        lead_sponsor_class = lead_sponsor.get('class')
        
        # Extract collaborators (list)
        collaborators = sponsors.get('collaborators', [])
        collaborator_names = [c.get('name') for c in collaborators if c.get('name')]
        collaborator_classes = [c.get('class') for c in collaborators if c.get('class')]
        num_collaborators = len(collaborators)
        num_collaborators_plus_lead = num_collaborators + 1 if lead_sponsor_name else num_collaborators
        
        # Extract responsible party / overall officials
        responsible_party = sponsors.get('responsibleParty', {})
        responsible_party_investigator_full_name = responsible_party.get('investigatorFullName')
        responsible_party_investigator_affiliation = responsible_party.get('investigatorAffiliation')
        
        # Extract overall officials
        contacts_locations = protocol.get('contactsLocationsModule', {})
        overall_officials = contacts_locations.get('overallOfficials', [])
        
        overall_official_names = [o.get('name') for o in overall_officials if o.get('name')]
        overall_official_affiliations = [o.get('affiliation') for o in overall_officials if o.get('affiliation')]
        overall_official_roles = [o.get('role') for o in overall_officials if o.get('role')]
        
        # Extract conditions and interventions
        conditions_module = protocol.get('conditionsModule', {})
        conditions = conditions_module.get('conditions', [])
        
        arms_interventions = protocol.get('armsInterventionsModule', {})
        interventions = arms_interventions.get('interventions', [])
        intervention_names = [i.get('name') for i in interventions if i.get('name')]
        intervention_types = [i.get('type') for i in interventions if i.get('type')]
        
        # Extract design information
        design = protocol.get('designModule', {})
        study_type = design.get('studyType')
        phases = design.get('phases', [])
        primary_purpose = safe_get(design, 'designInfo', 'primaryPurpose')
        
        # Extract enrollment
        enrollment_info = design.get('enrollmentInfo', {})
        enrollment_count = enrollment_info.get('count')
        
        # Extract primary outcome
        outcomes = protocol.get('outcomesModule', {})
        primary_outcomes = outcomes.get('primaryOutcomes', [])
        primary_outcome_measures = [p.get('measure') for p in primary_outcomes if p.get('measure')]
        
        # Extract locations
        locations = contacts_locations.get('locations', [])
        num_locations = len(locations)
        
        location_facilities = [loc.get('facility') for loc in locations if loc.get('facility')]
        location_cities = [loc.get('city') for loc in locations if loc.get('city')]
        location_states = [loc.get('state') for loc in locations if loc.get('state')]
        location_countries = [loc.get('country') for loc in locations if loc.get('country')]
        location_statuses = [loc.get('status') for loc in locations if loc.get('status')]
        
        # Extract geopoints
        geopoints = [loc.get('geoPoint') for loc in locations if loc.get('geoPoint')]
        
        # Extract MeSH terms from derived section
        derived = api_response.get('derivedSection', {})
        condition_browse = derived.get('conditionBrowseModule', {})
        condition_mesh_terms = [m.get('term') for m in condition_browse.get('meshes', []) if m.get('term')]
        
        intervention_browse = derived.get('interventionBrowseModule', {})
        intervention_mesh_terms = [m.get('term') for m in intervention_browse.get('meshes', []) if m.get('term')]
        
        # Extract has results
        has_results = api_response.get('hasResults', False)
        
        # Extract oversight
        oversight = protocol.get('oversightModule', {})
        oversight_has_dmc = oversight.get('oversightHasDmc')
        is_fda_regulated_drug = oversight.get('isFdaRegulatedDrug')
        is_fda_regulated_device = oversight.get('isFdaRegulatedDevice')
        
        # Extract references/citations
        references_module = protocol.get('referencesModule', {})
        references = references_module.get('references', [])
        citations = []
        pmids = []
        for ref in references:
            citations.append(ref.get('citation'))
            pmids.append(ref.get('pmid'))
        
        # Build flattened dictionary
        result = {
            # Basic identification
            'nct_id': nct_id,
            'brief_title': brief_title,
            'official_title': official_title,
            'overall_status': overall_status,
            
            # Organization & Sponsor 
            'lead_sponsor_name': lead_sponsor_name,
            'lead_sponsor_class': lead_sponsor_class,
            'collaborator_names': collaborator_names,  
            'collaborator_classes': collaborator_classes, 
            'org_full_name': org_full_name,
            
            # Key personnel 
            'overall_official_names': overall_official_names, 
            'overall_official_affiliations': overall_official_affiliations,  
            'overall_official_roles': overall_official_roles,  
            'responsible_party_investigator_full_name': responsible_party_investigator_full_name,
            'responsible_party_investigator_affiliation': responsible_party_investigator_affiliation,
            'num_collaborators': num_collaborators,
            
            # Scientific focus 
            'conditions': conditions, 
            'intervention_names': intervention_names, 
            'intervention_types': intervention_types,  
            'phases': phases,  
            'primary_outcome_measures': primary_outcome_measures, 
            
            # Study scope & capacity 
            'enrollment_count': enrollment_count,
            'study_type': study_type,
            'num_locations': num_locations,
            'location_facilities': location_facilities, 
            'location_cities': location_cities,  
            'location_states': location_states, 
            'location_countries': location_countries, 
            
            # Experience & track record 
            'study_first_post_date': study_first_post_date,
            'completion_date': completion_date,
            'has_results': has_results,
            'num_collaborators_plus_lead': num_collaborators_plus_lead,
            
            # Therapeutic area expertise
            'condition_mesh_terms': condition_mesh_terms, 
            'intervention_mesh_terms': intervention_mesh_terms, 
            'primary_purpose': primary_purpose,
            
            # Current activity status 
            'last_update_post_date': last_update_post_date,
            'recruitment_status': recruitment_status,
            'start_date': start_date,
            'primary_completion_date': primary_completion_date,
            
            # Secondary fields 
            'oversight_has_dmc': oversight_has_dmc,
            'is_fda_regulated_drug': is_fda_regulated_drug,
            'is_fda_regulated_device': is_fda_regulated_device,
            'location_statuses': location_statuses,  
            
            # Additional fields
            'citations': citations,  
            'pmids': pmids,  
            'geopoints': geopoints 
        }
        
        return result


    def forward(self, query_cond:str=None, query_term:str=None,query_lead:str=None,max_results: int = 5000) -> list:
        """
        Search ClinicalTrials.gov for trials.
        
        Args:
            query_cond: Disease or condition (e.g., 'lung cancer', 'diabetes')
            query_term : Other terms (e.g., 'AREA[LastUpdatePostDate]RANGE[2023-01-15,MAX]').
            query_lead : Searches the LeadSponsorName
            max_results: Number of trials to return (max: 1000)
        
        Returns:
            List of strings, each string being a structured representation of a trial.
        """
        
        params = {
            "query.cond": query_cond,
            "query.term":query_term,
            "query.lead":query_lead,
            "pageSize": min(max_results, 5000),
            "format": "json"
        }
        params = {k: v for k, v in params.items() if v is not None}
        try:
            response = requests.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            studies = response.json().get("studies", [])
            
            structured_trials = []
            for i, study in enumerate(studies):
                structured_data = self._extract_partner_info(study)
                structured_trials.append(structured_data)

            return structured_trials
            
        except Exception as e:
            return [f"Error searching clinical trials: {str(e)}"]
        

