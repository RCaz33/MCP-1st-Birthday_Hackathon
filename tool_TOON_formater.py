def TOON_formater(api_response):
    """
    Extract core partner identification information from ClinicalTrials.gov API response.
    
    Args:
        api_response (dict): Raw API response from ClinicalTrials.gov
        
    Returns:
        str: TOOn (Token-Oriented Object Notation) formatted string with 41 core fields
    """
    
    # Helper function to safely navigate nested dicts
    def safe_get(data, *keys, default=None):
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, {})
            else:
                return default
        return data if data != {} else default
    
    # Helper function to format value for TOOn
    def format_value(val):
        if val is None:
            return ''
        elif isinstance(val, bool):
            return str(val).lower()
        else:
            return str(val)
    
    # Helper function to format list for TOOn
    def format_list(lst):
        if not lst:
            return ''
        # Escape commas in individual items by wrapping in quotes if needed
        formatted_items = []
        for item in lst:
            item_str = format_value(item)
            if ',' in item_str or '\n' in item_str:
                item_str = f'"{item_str}"'
            formatted_items.append(item_str)
        return ','.join(formatted_items)
    
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
    recruitment_status = overall_status
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
    
    # Extract responsible party
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
    
    # Extract MeSH terms
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
    
    # Build TOOn formatted output
    toon_lines = []
    
    # Basic identification
    toon_lines.append(f"nct_id: {format_value(nct_id)}")
    toon_lines.append(f"brief_title: {format_value(brief_title)}")
    toon_lines.append(f"official_title: {format_value(official_title)}")
    toon_lines.append(f"overall_status: {format_value(overall_status)}")
    
    # Organization & Sponsor
    toon_lines.append(f"lead_sponsor_name: {format_value(lead_sponsor_name)}")
    toon_lines.append(f"lead_sponsor_class: {format_value(lead_sponsor_class)}")
    toon_lines.append(f"collaborator_names[{len(collaborator_names)}]: {format_list(collaborator_names)}")
    toon_lines.append(f"collaborator_classes[{len(collaborator_classes)}]: {format_list(collaborator_classes)}")
    toon_lines.append(f"org_full_name: {format_value(org_full_name)}")
    
    # Key personnel
    toon_lines.append(f"overall_official_names[{len(overall_official_names)}]: {format_list(overall_official_names)}")
    toon_lines.append(f"overall_official_affiliations[{len(overall_official_affiliations)}]: {format_list(overall_official_affiliations)}")
    toon_lines.append(f"overall_official_roles[{len(overall_official_roles)}]: {format_list(overall_official_roles)}")
    toon_lines.append(f"responsible_party_investigator_full_name: {format_value(responsible_party_investigator_full_name)}")
    toon_lines.append(f"responsible_party_investigator_affiliation: {format_value(responsible_party_investigator_affiliation)}")
    toon_lines.append(f"num_collaborators: {format_value(num_collaborators)}")
    
    # Scientific focus
    toon_lines.append(f"conditions[{len(conditions)}]: {format_list(conditions)}")
    toon_lines.append(f"intervention_names[{len(intervention_names)}]: {format_list(intervention_names)}")
    toon_lines.append(f"intervention_types[{len(intervention_types)}]: {format_list(intervention_types)}")
    toon_lines.append(f"phases[{len(phases)}]: {format_list(phases)}")
    toon_lines.append(f"primary_outcome_measures[{len(primary_outcome_measures)}]: {format_list(primary_outcome_measures)}")
    
    # Study scope & capacity
    toon_lines.append(f"enrollment_count: {format_value(enrollment_count)}")
    toon_lines.append(f"study_type: {format_value(study_type)}")
    toon_lines.append(f"num_locations: {format_value(num_locations)}")
    toon_lines.append(f"location_facilities[{len(location_facilities)}]: {format_list(location_facilities)}")
    toon_lines.append(f"location_cities[{len(location_cities)}]: {format_list(location_cities)}")
    toon_lines.append(f"location_states[{len(location_states)}]: {format_list(location_states)}")
    toon_lines.append(f"location_countries[{len(location_countries)}]: {format_list(location_countries)}")
    
    # Experience & track record
    toon_lines.append(f"study_first_post_date: {format_value(study_first_post_date)}")
    toon_lines.append(f"completion_date: {format_value(completion_date)}")
    toon_lines.append(f"has_results: {format_value(has_results)}")
    toon_lines.append(f"num_collaborators_plus_lead: {format_value(num_collaborators_plus_lead)}")
    
    # Therapeutic area expertise
    toon_lines.append(f"condition_mesh_terms[{len(condition_mesh_terms)}]: {format_list(condition_mesh_terms)}")
    toon_lines.append(f"intervention_mesh_terms[{len(intervention_mesh_terms)}]: {format_list(intervention_mesh_terms)}")
    toon_lines.append(f"primary_purpose: {format_value(primary_purpose)}")
    
    # Current activity status
    toon_lines.append(f"last_update_post_date: {format_value(last_update_post_date)}")
    toon_lines.append(f"recruitment_status: {format_value(recruitment_status)}")
    toon_lines.append(f"start_date: {format_value(start_date)}")
    toon_lines.append(f"primary_completion_date: {format_value(primary_completion_date)}")
    
    # Secondary fields
    toon_lines.append(f"oversight_has_dmc: {format_value(oversight_has_dmc)}")
    toon_lines.append(f"is_fda_regulated_drug: {format_value(is_fda_regulated_drug)}")
    toon_lines.append(f"is_fda_regulated_device: {format_value(is_fda_regulated_device)}")
    toon_lines.append(f"location_statuses[{len(location_statuses)}]: {format_list(location_statuses)}")
    
    # Additional fields
    toon_lines.append(f"citations[{len(citations)}]: {format_list(citations)}")
    toon_lines.append(f"pmids[{len(pmids)}]: {format_list(pmids)}")
    
    # Geopoints (structured data - format as array of objects)
    if geopoints:
        geo_keys = set()
        for gp in geopoints:
            if gp:
                geo_keys.update(gp.keys())
        
        if geo_keys:
            geo_keys_sorted = sorted(geo_keys)
            toon_lines.append(f"geopoints[{len(geopoints)}]{{{','.join(geo_keys_sorted)}}}:")
            for gp in geopoints:
                if gp:
                    values = [format_value(gp.get(k)) for k in geo_keys_sorted]
                    toon_lines.append(f"  {','.join(values)}")
                else:
                    toon_lines.append(f"  {','.join(['' for _ in geo_keys_sorted])}")
    else:
        toon_lines.append(f"geopoints[0]:")
    
    return '\n'.join(toon_lines)