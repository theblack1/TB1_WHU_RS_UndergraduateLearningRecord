
package ch04.usefulclass;

public class CharacterDemo202 {

	public static void main(String[] args) {

		// 创建数值为'A'的Character对象
		Character objChar1 = new Character('A');
		// 从Character对象返回char值
		char ch = objChar1.charValue();

		// 字符串比较
		Character objChar2 = new Character('C');
		int result = objChar1.compareTo(objChar2);
		// result = -2，表示objChar1小于objChar2
		if (result < 0) {
			System.out.println("objChar1小于objChar2");
		}
	}
}
