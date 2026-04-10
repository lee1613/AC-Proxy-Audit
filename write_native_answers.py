import json

with open("eval_dataset.json", "r") as f:
    eval_data = json.load(f)

# Manually populated by the Agent natively reasoning through contexts_dump.json
answers_map = {
    "1": {"answer": "No information found", "source": "N/A"},
    "2": {"answer": "No information found", "source": "N/A"},
    "3": {"answer": "No information found", "source": "N/A"},
    "4": {"answer": "Any named user Monte Carlo account that hasn't logged for more than 90 days will be deactivated.", "source": "enterprise-data/data-governance/data-management.md"},
    "5": {"answer": "No information found", "source": "N/A"},
    "6": {"answer": "No information found", "source": "N/A"},
    "7": {"answer": "No information found", "source": "N/A"},
    "8": {"answer": "No information found", "source": "N/A"},
    "9": {"answer": "These credentials are secured out-of-band in a vault that requires declaring an incident and requires Identity team two person rule approval to get access to ensure compliance.", "source": "security/identity/gitops/okta/_index.md"},
    "10": {"answer": "Role-based access control (RBAC) is used for separation of duties. In competitive products roles are broader and when a person changes roles, their permissions must be changed manually. You want permissions to change automatically to avoid insider threats.", "source": "marketing/brand-and-product-marketing/product-and-solution-marketing/usecase-gtm/devsecops/_index.md"},
    "11": {"answer": "No information found", "source": "N/A"},
    "12": {"answer": "Designated User Override allowed specific users, groups, and custom roles to bypass merge request approval policies during critical situations while maintaining comprehensive audit trails and governance controls.", "source": "engineering/architecture/design-documents/approval_policies_bypass/_index.md"},
    "13": {"answer": "Tableau access is structured in a tiered approach that separates data based on sensitivity levels and regulatory requirements. General Content allows all users, whereas Restricted SAFE Content requires additional approval and justification.", "source": "enterprise-data/platform/tableau/_index.md"},
    "14": {"answer": "No information found", "source": "N/A"},
    "15": {"answer": "No information found", "source": "N/A"},
    "16": {"answer": "No information found", "source": "N/A"},
    "17": {"answer": "Data of different types MUST be logically seperated at rest. Virtual networks (for example, VPC in GCP) MAY be used as a mechanism for data and workload isolation.", "source": "security/planning/security-development-deployment-requirements.md"},
    "18": {"answer": "No information found", "source": "N/A"},
    "19": {"answer": "No information found", "source": "N/A"},
    "20": {"answer": "No information found", "source": "N/A"},
    "21": {"answer": "No information found", "source": "N/A"},
    "22": {"answer": "No information found", "source": "N/A"},
    "23": {"answer": "Ensure ingested data does not contain sensitive information and anonymize, pseudonymize, or redact data as needed.", "source": "engineering/architecture/design-documents/gitlab_messaging_layer/_index.md"},
    "24": {"answer": "No information found", "source": "N/A"},
    "25": {"answer": "Run service processes as their own users with exactly the set of privileges they require, and give only the minimum level of access rights that is necessary to complete an assigned operation.", "source": "security/product-security/security-platforms-architecture/security-architecture/_index.md"},
    "26": {"answer": "Predefined security groups are utilized to assign role-based access privileges and segregate access to data. Administrator access is granted based on job roles and responsibilities and limited to authorized personnel.", "source": "security/security-assurance/technical-and-organizational-measures.md"},
    "27": {"answer": "No information found", "source": "N/A"},
    "28": {"answer": "Administrator access to the production systems is granted based on job roles and responsibilities and limited to authorized personnel. Privileges flow through the user role, with exceptions for privileged roles such as accountadmin, securityadmin, useradmin, and sysadmin.", "source": "enterprise-data/platform/_index.md"},
    "29": {"answer": "No information found", "source": "N/A"},
    "30": {"answer": "No information found", "source": "N/A"},
    "31": {"answer": "Repeated failed login attempts need to trigger a temporary account lockout after 10 failed attempts, or the minimum value allowed by the system.", "source": "security/policies_and_standards/password-standard.md"},
    "32": {"answer": "No information found", "source": "N/A"},
    "33": {"answer": "No information found", "source": "N/A"},
    "34": {"answer": "All audit logs for the super administrator event log actions are exported by default to the centralized logging system.", "source": "security/identity/gitops/okta/_index.md"},
    "35": {"answer": "Workspaces are terminated automatically after a given period of time in order to avoid unnecessary resource usage and costs.", "source": "engineering/architecture/design-documents/workspaces/decisions/006_automatic_termination_of_workspace.md"},
    "36": {"answer": "No information found", "source": "N/A"},
    "37": {"answer": "No information found", "source": "N/A"},
    "38": {"answer": "No information found", "source": "N/A"},
    "39": {"answer": "No information found", "source": "N/A"},
    "40": {"answer": "No information found", "source": "N/A"},
    "41": {"answer": "Virtual private networks (VPN) are used to allow access to less secured resources, typically also protected by an enterprise firewall.", "source": "security/product-security/security-platforms-architecture/security-architecture/zero-trust.md"},
    "42": {"answer": "All external communication MUST be encrypted in transit using up to date protocals and ciphers. All internal communication SHOULD be encrypted in transit if possible.", "source": "security/planning/security-development-deployment-requirements.md"},
    "43": {"answer": "All devices must have Full Disk Encryption (FDE/FileVault) enabled.", "source": "security/corporate/systems/_index.md"},
    "44": {"answer": "Vendor will maintain industry standard best practices for preventing the unauthorized access, alteration, or removal of data during transfer. This includes encryption of data at rest in production datastores using strong industry standard encryption algorithms, and encryption of data in transit using industry standard protocols.", "source": "finance/procurement/vendor-guidelines/vendor-security-addendum.md"},
    "45": {"answer": "Established review and approval processes for any access requests to services storing customer Data or systems and applications that are in scope for the services provided under the Agreement.", "source": "finance/procurement/vendor-guidelines/vendor-security-addendum.md"},
    "46": {"answer": "Every vendor that handles personal data is required to go through a Privacy Review prior to being onboarded, which includes completion and approval of the privacy due diligence questionnaires detailed in the Procurement process.", "source": "legal/privacy/_index.md"},
    "47": {"answer": "Always apply dynamic masking policies to personal data fields in the PREP and PROD layers of Snowflake.", "source": "enterprise-data/data-governance/data-management.md"},
    "48": {"answer": "Access requests are approved prior to making access changes. (AC-2)", "source": "security/security-and-technology-policies/access-management-policy.md"},
    "49": {"answer": "No information found", "source": "N/A"},
    "50": {"answer": "No information found", "source": "N/A"}
}

# The ambiguous statements list is empty based on the agent's analysis
ambiguous = []
with open("ambiguous_statements.json", "w") as f:
    json.dump(ambiguous, f, indent=4)

updated_questions = []

for q in eval_data["audit_questions"]:
    qid = str(q["id"])
    new_q = {
        "id": q["id"],
        "family": q["family"],
        "control_id": q["control_id"],
        "question": q["question"]
    }
    ans = answers_map.get(qid, {"answer": "No information found", "source": "N/A"})
    new_q["answer"] = ans["answer"]
    new_q["source"] = ans["source"]
    updated_questions.append(new_q)

with open("eval_dataset.json", "w") as f:
    json.dump({"audit_questions": updated_questions}, f, indent=4)

print("eval_dataset.json and ambiguous_statements.json successfully rewritten natively by python mapping.")
