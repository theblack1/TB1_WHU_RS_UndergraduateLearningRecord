import java.util.Scanner;

public class Division {
	public static void main(String[] arges) {
		Scanner reader = new Scanner(System.in);
		System.out.println("Please input a number:");
		int num = 0;
		int digit = 0;
		int val = 0;
		num = Integer.valueOf(reader.nextInt());
		while (num / (int) Math.pow(10, digit - 1) != 0) {
			digit++;
		}
		for (int i = digit - 1; i > 0; i--) {
			val = num / (int) Math.pow(10, i);
			System.out.println(val);
			num -= val * (int) Math.pow(10, i);
		}
	}

}