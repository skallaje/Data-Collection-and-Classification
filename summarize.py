"""
sumarize.py
"""
def log_project_summary():
    f = open("cs579_project_summary.txt", "w")
    f.write("SUMMARY OF COLLECTION, CLUSTERING, CLASSIFICATION\n")
    for tempfile in [open("collection_summary.txt","r"), open("clustering_summary.txt","r"), open("classification_summary.txt","r")]:
        f.write("\n")
        f.write(tempfile.read())

def main():
    log_project_summary()
    
if __name__ == '__main__':
    main()