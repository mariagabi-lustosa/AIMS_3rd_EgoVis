from notebooks.utils import package_results_for_submission_ek100

CFG_FILES = [
    ('expts/02_ek100_avt_tsn_test_testonly.txt', 0),
]
WTS = [1.0]
SLS = [1, 4, 4]

package_results_for_submission_ek100(CFG_FILES, WTS, SLS)


#uids = f["uid"][:]
#logits = f["logits/action"][:]

#submission = {}

#for uid, logit in zip(uids, logits):
#    if isinstance(uid, bytes):
#        uid = uid.decode("utf-8")
#    else:
#        uid = str(uid)

#    top5 = np.argsort(logit)[-5:][::-1]
#    submission[uid] = top5.tolist()

#with open("submission.json", "w") as f_json:
#    json.dump(submission, f_json)

#print("✅ submission.json criado!")