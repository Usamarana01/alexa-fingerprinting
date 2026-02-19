import pandas as pd
import os

def test_append():
    central_file = "collection_test.csv"
    if os.path.exists(central_file):
        os.remove(central_file)
        
    # Mock some data
    data1 = pd.DataFrame({
        'time': [0.1, 0.2],
        'size': [100, 200],
        'direction': [1.0, -1.0]
    })
    
    # Simulate first save
    safe_name = "test_site"
    repeat = 1
    label = f"{safe_name}_repeat_{repeat}"
    append_df = data1.copy()
    append_df.insert(0, 'label', label)
    
    header = not os.path.exists(central_file) or os.path.getsize(central_file) == 0
    append_df.to_csv(central_file, mode='a', index=False, header=header)
    
    # Simulate second save
    data2 = pd.DataFrame({
        'time': [0.3],
        'size': [300],
        'direction': [1.0]
    })
    repeat = 2
    label = f"{safe_name}_repeat_{repeat}"
    append_df = data2.copy()
    append_df.insert(0, 'label', label)
    
    header = not os.path.exists(central_file) or os.path.getsize(central_file) == 0
    append_df.to_csv(central_file, mode='a', index=False, header=header)
    
    # Verify
    result = pd.read_csv(central_file)
    print("Verification result:")
    print(result)
    
    assert len(result) == 3
    assert 'label' in result.columns
    assert result['label'].iloc[0] == "test_site_repeat_1"
    assert result['label'].iloc[2] == "test_site_repeat_2"
    print("\n✅ Verification successful!")
    
    os.remove(central_file)

if __name__ == "__main__":
    test_append()
