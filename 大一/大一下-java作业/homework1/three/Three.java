import java.util.*;

public class Three{
	public static void main(String[]args){
		Scanner reader = new Scanner(System. in);
		System.out.print("WHO ARE YOU? ");
		String name = reader.next();
		System.out.print("WHERE DO YOU LREAN JAVA? ");
		String where = reader.next();
		System.out.print("HOW WOULD YOU LIKE TO LEARN IT? ");
		String how = reader.next();
		System.out.println("Hello," + name + "\n" + "You plan to learn java " + where + "\n" + "And you will learn it by " + how);
	}
	

}