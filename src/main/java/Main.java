import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.ndarray.*;
import ai.djl.translate.*;
import java.nio.file.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Path modelDir = Paths.get("model");
        Model model = Model.newInstance("model/model.pt");
        model.load(modelDir);

        Translator<Float, Float> translator = new Translator<Float, Float>(){

            @Override
            public NDList processInput(TranslatorContext ctx, Float input) {
                NDManager manager = ctx.getNDManager();
                NDArray array = manager.create(new float[] {input});
                return new NDList (array);
            }

            @Override
            public Float processOutput(TranslatorContext ctx, NDList list) {
                NDArray temp_arr = list.get(0);
                return temp_arr.getFloat();
            }

            @Override
            public Batchifier getBatchifier() {
                // The Batchifier describes how to combine a batch together
                // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
                return Batchifier.STACK;
            }
        };
        Predictor<Float, Float> predictor = model.newPredictor(translator);

        System.out.println(predictor.predict(5.0f));
    }
}