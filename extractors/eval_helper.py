class Note:
    def __init__(self,pitch,onset,duration):
        self.pitch=pitch
        self.onset=onset
        self.duration=duration


def diff_statistics(ground_truth_notes,estimated_notes):
    ground_truth_onset_pitch=set([(note.onset,note.pitch) for note in ground_truth_notes])
    estimated_onset_pitch=set([(note.onset,note.pitch) for note in estimated_notes])
    correct=len(ground_truth_onset_pitch & estimated_onset_pitch)
    true_positive=correct
    false_positive=len(estimated_onset_pitch)-correct
    false_negative=len(ground_truth_onset_pitch)-correct
    return true_positive,false_positive,false_negative

if __name__ == '__main__':
    ground_truth_notes=[Note(60,2,1),Note(65,2,2),Note(67,8,8)]
    estimated_notes=[Note(48,2,1),Note(65,2,3),Note(67,7,8),Note(60,10,2)]
    tp,fp,fn=diff_statistics(ground_truth_notes,estimated_notes)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*precision*recall/(precision+recall)
    print('Song level evaluation: P=%.2f%%, R=%.2f%%, F1=%.2f%%'%(precision*100,recall*199,f1*199))
