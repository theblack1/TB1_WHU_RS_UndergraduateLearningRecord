package ch04.usefulclass;

public class ObjectDemo101 {

	public static void main(String[] args) {

		Person101 person = new Person101("Tony", 18);
		// 打印过程自动调用person的 toString()方法
		System.out.println(person);
	}
}
