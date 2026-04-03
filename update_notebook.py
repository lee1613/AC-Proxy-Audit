import nbformat

def add_cell():
    nb_path = 'Data Cleaning.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Find the execution index where the NIST data is originally fetched/printed
    target_idx = -1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and ('print(nist[0])' in cell.source or 'nist = fetch_nist_txt()' in cell.source):
            target_idx = i

    new_code = '''# Isolate the Access Control (AC) section into a new variable
nist_ac_text = [policy.strip() for policy in nist if policy.strip().startswith(('AC-', '4.1 Access Control'))]

# View the result to ensure it captured the Access Control data
print(f"Isolated {len(nist_ac_text)} Access Control policies.")
print("Sample Preview:\\n", nist_ac_text[0][:150], "...")'''

    new_cell = nbformat.v4.new_code_cell(source=new_code)
    
    if target_idx != -1:
        nb.cells.insert(target_idx + 1, new_cell)
    else:
        nb.cells.append(new_cell)

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Successfully injected the new cell into the Notebook!")

if __name__ == "__main__":
    add_cell()
