import pandas as pd
import pickle
from common import get_embedding, chunk_text

def build_vector_database(excel_file, output_file):
    """Build vector database from Excel file and save to local file"""
    df = pd.read_excel(excel_file)
    processed_data = []
    for _, row in df.iterrows():
        chunks = chunk_text(row['name']+row['key'], 100)
        video_embeddings = []
        i_chunk = 0
        for chunk in chunks:
            i_chunk += 1
            embedding = get_embedding(chunk)
            video_embeddings.append(embedding)
            # print(f"row:{row['num']}-{chunk} -- chunk_index:{i_chunk}")
            print(f"row:{row['id']} -- chunk_index:{i_chunk}")
            print("--------------")
        processed_data.append({
            'video_id': row['id'],
            'video_name': row['name']+"_"+row['key'],
            'video_sub': row['sub']+"_"+row['grade']+"_"+row['vol']+"_"+row['unit'],
            # 'video_link': row['videoUrl'],
            # 'video_cover_link': row['cover'],
            # 'video_vid': row['vid'],
            # 'video_summary': row['summary'],
            'embeddings': video_embeddings
        })
       
    # ab+ 最佳
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Vector database saved to {output_file}")

if __name__ == "__main__":
    excel_file = "hjb-sx-1.1.xlsx"
    vector_db_file = "hjb-sx-1.1_vector_database.pkl"
    print(f"Building vector database from {excel_file} to {vector_db_file}")
    build_vector_database(excel_file, vector_db_file)