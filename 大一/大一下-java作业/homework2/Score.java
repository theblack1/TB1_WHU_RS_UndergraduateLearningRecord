import java.util.*;

public class Score{
	public static void main(String[] args){
		int score = 0;
		Scanner reader = new Scanner (System.in);
		System.out.println("What is your score?\n");
		score = Integer.valueOf(reader.nextInt());
		if (score<=100 && score >=90)
			System.out.println("You got an A!");
		else if (score<90 && score >=80)
			System.out.println("You got an B!");
		else if (score<80 && score >=70)
			System.out.println("You got an C!");
		else if (score<70 && score >=60)
			System.out.println("You got an D!");
		else if (score<60 && score >=0)
			System.out.println("You got an F!");
		else
			System.out.println("ERROR!!!");
	}
	
}