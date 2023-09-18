public class Factorials{
	public static void main(String[] args){
		int i = 1;
		double sum = 1;
		
		for (i = 1,sum = 1;i<=5;i++)
		{
			sum *= i;
		}
		System.out.println("5!=" + sum);
		
		for (i = 1,sum = 1;i<=10;i++)
		{
			sum *= i;
		}
		System.out.println("10!=" + sum);
		
		for (i = 1,sum = 1;i<=50;i++)
		{
			sum *= i;
		}
		System.out.println("50!=" + sum);
		
		for (i = 1,sum = 1;i<=100;i++)
		{
			sum *= i;
		}
		System.out.println("100!=" + sum);
		
	}
}