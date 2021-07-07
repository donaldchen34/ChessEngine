from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate

#363 - 90 Board Representation in bits "4.1 Feature Representation Giraffe"
#Missing 90 Features. Maybe convert some to more bit representation?
GLOBAL_FEATURES = 17
PIECE_CENTRIC_FEATURES = 160
SQUARE_CENTIC_FEATURES = 64
FEATURE_REPRESENTATION = GLOBAL_FEATURES + PIECE_CENTRIC_FEATURES + SQUARE_CENTIC_FEATURES
NUM = 3

POSSIBLE_MOVES = 4672 # 8x8x(8x7+8+9) -Change because no underpromotions
class DQN:
    def __init__(self,board):
        self.board = board
        pass

    #https://stackoverflow.com/questions/55233377/keras-sequential-model-with-multiple-inputs/55234203
    def createModel(self):

        global_features_input = Input(shape= GLOBAL_FEATURES,)
        x = Dense(GLOBAL_FEATURES * NUM, activation= 'relu')(global_features_input)
        piece_centric_input = Input(shape=PIECE_CENTRIC_FEATURES,)
        y = Dense(PIECE_CENTRIC_FEATURES * NUM, activation= 'relu')(piece_centric_input)
        square_centric_input = Input(shape= SQUARE_CENTIC_FEATURES,)
        z = Dense(SQUARE_CENTIC_FEATURES * NUM, activation= 'relu')(square_centric_input)
        merged = Concatenate(axis= 1)([x,y,z])
        t = Dense(FEATURE_REPRESENTATION,input_dim=3, activation= 'relu')(merged)
        output = Dense(POSSIBLE_MOVES, activation='tanh')(t)

        model = Model(inputs=[global_features_input,piece_centric_input,square_centric_input],
                      outputs=output)

        #model.compile()

        return model