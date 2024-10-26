import os
import zipfile

# Define the folder that contains the weekly files
output_dir = "weekly_feat_commits"
zip_filename = "weekly_feat_commits.zip"

# Step 5: Zip the directory
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=os.path.join(os.path.basename(root), file))

print(f"Folder '{output_dir}' has been zipped as '{zip_filename}'")
