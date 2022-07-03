import me.mikex86.scicore.DataType;
import me.mikex86.scicore.ITensor;
import me.mikex86.scicore.SciCore;

public class Test {

    public static void main(String[] args) {
        SciCore sc = new SciCore();
        sc.setBackend(SciCore.BackendType.JVM);
        ITensor a = sc.random(DataType.FLOAT32, 3, 2);
        ITensor b = sc.random(DataType.FLOAT32, 2, 3);
        ITensor c = sc.matmul(a, b);
        System.out.println("A: \n" + a);
        System.out.println("B: \n" + b);
        System.out.println("Result: \n" + c);
    }

}
