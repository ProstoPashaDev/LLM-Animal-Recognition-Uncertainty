def find_animal_confidense(answer):
    if answer.count("\n") > 1:
        ans1 = answer.split("\n")[0]
        ans2 = answer.split("\n")[-1]
        if len(ans1.split(" ")) == 2:
            res = ans1
        elif len(ans2.split(" ")) == 2:
            res = ans2
        else:
            dig = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}
            l = 0
            r = 0
            for i in range(len(answer)):
                if answer[i] in dig:
                    l = i - 2
                    r = i
                    while r < len(answer) and (
                            answer[r] != " " and answer[r] != "." and answer[r] != "\n" and answer[r] != "*"):
                        r += 1

                    while l > 0 and (answer[l] != " " and answer[l] != "." and answer[l] != "\n" and answer[l] != "*"):
                        l -= 1

                    break

            res = answer[l + 1:r]
    else:
        res = answer
    return res

def split_animal_confidence(answer):
    try:
        animal_ans = answer.split(",")[0]
        conf = answer.split(",")[1]
    except:
        animal_ans = answer.split(" ")[0]
        conf = answer.split(" ")[1]

    if conf[-1] == "%":
        conf = conf[:len(conf) - 1:]
    return animal_ans, conf


def print_image_dataset(dataset):
    for key, value in dataset.items():
        print(key, value.split("\\")[-1])
