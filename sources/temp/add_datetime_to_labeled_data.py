import pandas as pd
from sources.utils import *

def remove_reply(value):
    if not value or not isinstance(value, str):
        return value
    lines = value.split("\n")
    rem_reply_lines = [line for line in lines if
                       len(line) > 0 and line[0] != '>' and not line.startswith('(In reply to')]

    return '\n'.join(rem_reply_lines)

def normalize_text(text):
    return " ".join(str(text).split())

if __name__ == "__main__":
    labeled_data = pd.read_csv(mapped_comment_data)

    print(f"Number of comments in labeled data: {len(labeled_data)}")

    #read json file
    with open(issue_data_path, 'r') as f:
        comment_data = json.load(f)

    count = 0
    for issue_id in labeled_data['issue_id'].unique():
        labeled_issue_data = labeled_data[labeled_data['issue_id'] == issue_id]
        issue_comment_with_time = [i for i in comment_data if i['id'] == issue_id][0]

        for index, row in labeled_issue_data.iterrows():
            for comment in issue_comment_with_time['comments']:
                text_to_match = normalize_text(remove_reply(row['text']))

                text_with_time_text = normalize_text(remove_reply(comment['text']))
                text_with_time_rawtext = normalize_text(remove_reply(comment['raw_text']))

                if text_to_match in text_with_time_text or text_to_match in text_with_time_rawtext:
                    labeled_data.at[index, 'creation_time'] = comment['creation_time']
                    labeled_data.at[index, 'comment_id'] = comment['id']
                    count += 1
                    break

    print(f"Number of comments matched: {count}")

    for index, row in labeled_data.iterrows():
        if pd.isnull(row['creation_time']):
            print(f"Comment not matched: {row['text']}")

    print("Finished adding datetime to labeled data.")

    def check_data_modified(mapped_comment_data, mapped_comment_data_with_time):
        data1 = pd.read_csv(mapped_comment_data)
        data2 = pd.read_csv(mapped_comment_data_with_time)

        for i in range(len(data1)):
            if data1.iloc[i]['issue_id'] != data2.iloc[i]['issue_id'] \
                or data1.iloc[i]['text_id'] != data2.iloc[i]['text_id'] \
                or data1.iloc[i]['text'] != data2.iloc[i]['text'] \
                or data1.iloc[i]['code'] != data2.iloc[i]['code'] \
                or data1.iloc[i]['label'] != data2.iloc[i]['label']:
                
                print(f"Data modified at index {i}:")
                print(f"Original text: {data1.iloc[i]['text']}")
                print(f"Modified text: {data2.iloc[i]['text']}")
            

    check_data_modified(mapped_comment_data, mapped_comment_data_with_time)

    #save the updated labeled data to a new csv file
    labeled_data.to_csv(mapped_comment_data_with_time, index=False)
