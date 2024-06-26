import pandas as pd
import pickle
from common import get_embedding, chunk_text

def build_vector_database(excel_file, output_file):
    """Build vector database from Excel file and save to local file"""
    df = pd.read_excel(excel_file)
    processed_data = []
    for _, row in df.iterrows():
        chunks = chunk_text(row['sub']+row['summary'])
        video_embeddings = []
        i_chunk = 0
        for chunk in chunks:
            i_chunk += 1
            embedding = get_embedding(chunk)
            video_embeddings.append(embedding)
            # print(f"row:{row['num']}-{chunk} -- chunk_index:{i_chunk}")
            print(f"row:{row['num']} -- chunk_index:{i_chunk}")
            print("--------------")
        processed_data.append({
            'video_id': row['num'],
            'video_name': row['name'],
            'video_sub': row['grade']+"_"+row['sub']+"_"+row['volume'],
            'video_teacher': row['teacher']+"_"+row['school'],
            'video_link': row['videoUrl'],
            'video_cover_link': row['cover'],
            'video_vid': row['vid'],
            'video_summary': row['summary'],
            'embeddings': video_embeddings
        })
       
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Vector database saved to {output_file}")

if __name__ == "__main__":
    excel_file = "znd456.xlsx"
    vector_db_file = "znd456_vector_database.pkl"
    print(f"Building vector database from {excel_file} to {vector_db_file}")
    build_vector_database(excel_file, vector_db_file)