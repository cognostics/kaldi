This is the cognostics example addition to kaldi in which we bundle all the
scripts necessary to convert videos to text. 

Eventually we might want to combine all the training data we have downloaded
and train a super-potent speech recognizer. This would also be done here. As of
right now we are still using the model trained on the librispeech corpus with
the provided example.

Please be wary of the normal file structure for kaldi examples. It's basically
set in stone.

Possible TODOs:
        - It is actually possible to just pass the stream that converts .mp4 to
        wav on the fly to the wav.scp file instead of converting it beforehand,
        it might make sense to implement that (it has been more problematic in
                        the past, which is why I chose the conventional route,
                        i.e. I extract the wav first and then point to that
                        file).
